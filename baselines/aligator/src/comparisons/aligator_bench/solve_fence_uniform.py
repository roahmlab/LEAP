#!/usr/bin/env python3
"""One uniform cold-start Aligator solve of the fence benchmark for one robot.

Formulation (the acados_fence_accuracy parity statement): fixed 0.005 s
mesh, state (q, v), controls = four lifted RK4 accelerations + actuator
torques + contact forces, linear RK4 state propagation, and ONE
least-squares feasibility objective made of every original bound / contact /
collision / boundary row (positive-part residuals) plus the full Pinocchio
inverse-dynamics residual at each RK stage. Same analytic seed as acados
(benchmark seed + fixed 0.2 m flight lift), one native solver call, only the
returned final iterate is judged, by the same independent Pinocchio judge.

Only public Aligator APIs are used (StageModel / CostStack /
QuadraticResidualCost / LinearDiscreteDynamics / SolverFDDP or
SolverProxDDP). The installed library is not modified. The user residuals are
CasADi-generated C (GeneratedResidual) and a native Pinocchio RNEA model
(InverseRK4); a stock CallbackBase raises at the wall-clock cap so the sole
call returns its last accepted iterate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'comparisons'))
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_DYNAMIC'] = 'FALSE'

ROBOTS = ['hopper2d', 'biped5', 'go2', 'h1', 'g1_29dof', 'talos', 'apollo']


def pin_physical_cores(count, offset=0):
    """Pin this process to `count` distinct physical cores, skipping the
    first `offset` physical cores of the allowed set (so several solves can
    share a machine without sharing cores). One logical CPU per core."""
    seen, cores = set(), []
    for cpu in sorted(os.sched_getaffinity(0)):
        top = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((top / 'physical_package_id').read_text(), (top / 'core_id').read_text())
        if key not in seen:
            seen.add(key)
            cores.append(cpu)
    chosen = cores[offset:offset + count]
    if len(chosen) != count:
        raise RuntimeError(f'Requested {count} physical cores at offset {offset}; found {len(chosen)}')
    os.sched_setaffinity(0, chosen)
    return chosen


def row_names(built, model, nreg):
    """Name every row of the running / terminal residual expressions, in the
    order build_lifted + matrix_orientation produce them."""
    sym = built.sym
    nx, nuu = sym.nx, sym.nuu
    running = list(sym.h_names) + [f'x{i}' for i in range(nx)] + [f'u{i}' for i in range(nuu)]
    running += ['inv'] * built.inverse_dim + ['reg'] * nreg
    running += ['rotmat'] * int(built.matrix_rotation_expr.numel())
    terminal = [sym.h_names[i] for i in sym.xonly_idx] + [f'x{i}' for i in range(nx)]
    terminal += ['rotmat'] * int(built.matrix_rotation_terminal_expr.numel())
    assert len(running) == int(model.cost_y_expr.numel()), (len(running), model.cost_y_expr.numel())
    assert len(terminal) == int(model.cost_y_expr_e.numel()), (len(terminal), model.cost_y_expr_e.numel())
    return running, terminal


def foh_inverse_numeric(r, act, x, u, h):
    """Numeric RNEA at the four RK4 stage points with FOH torque/force."""
    import pinocchio as pin
    nv, nt = r.nv, r.nu
    nf = sum(c.dim for c in r.contacts)
    a = u[:4 * nv].reshape(4, nv)
    tau0, tau1 = u[4 * nv:4 * nv + nt], u[4 * nv + nt:4 * nv + 2 * nt]
    F0, F1 = u[4 * nv + 2 * nt:4 * nv + 2 * nt + nf], u[4 * nv + 2 * nt + nf:]
    q, v = x[:nv], x[nv:]
    d = r.data()
    points = [(q, v), (q + h * v / 2, v + h * a[0] / 2), (q + h * v / 2 + h * h * a[0] / 4, v + h * a[1] / 2),
              (q + h * v + h * h * a[1] / 2, v + h * a[2])]
    S = np.zeros((nv, nt)); S[act, np.arange(nt)] = 1.
    out = []
    for j, ((qq, vv), aa) in enumerate(zip(points, a)):
        c = [0., .5, .5, 1.][j]
        tau, F = (1 - c) * tau0 + c * tau1, (1 - c) * F0 + c * F1
        pin.computeJointJacobians(r.model, d, qq); pin.updateFramePlacements(r.model, d)
        generalized = S @ tau
        off = 0
        for ct in r.contacts:
            J = pin.getFrameJacobian(r.model, d, ct.frame_id, pin.LOCAL_WORLD_ALIGNED)[:ct.dim]
            generalized = generalized + J.T @ F[off:off + ct.dim]; off += ct.dim
        out.extend(pin.rnea(r.model, d, qq, vv, aa) - generalized)
    return np.asarray(out)


def family(name):
    if name.startswith(('pos_', 'rot_', 'vel_', 'mon_', 'keepout_', 'selfcol_')) or name == 'rotmat':
        return 'geometry'
    if name.startswith(('cone_', 'copA', 'copB', 'yaw')):
        return 'force'
    if name.startswith('x'):
        return 'state'
    if name.startswith('u'):
        return 'control'
    if name == 'inv' or name == 'reg':
        return name
    return name  # 'inv' / 'reg'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--robot', required=True, choices=ROBOTS)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cores', type=int, default=12)
    ap.add_argument('--core-offset', type=int, default=0)
    ap.add_argument('--dt', type=float, default=.005)
    ap.add_argument('--solver', choices=['fddp', 'proxddp'], default='fddp')
    ap.add_argument('--hold', choices=['zoh', 'foh'], default='foh',
                    help='torque/force hold inside each interval: zero-order (acados parity) or first-order with independent endpoints')
    ap.add_argument('--iterations', type=int, default=10000)
    ap.add_argument('--tol', type=float, default=1e-8)
    ap.add_argument('--time-limit', type=float, default=10800., help='Wall-clock cap [s] on the sole solver call')
    ap.add_argument('--reg-init', type=float, default=1e-3)
    ap.add_argument('--reg-min', type=float, default=1e-8)
    ap.add_argument('--reg-max', type=float, default=1e14)
    ap.add_argument('--mu-init', type=float, default=1e-3, help='ProxDDP AL penalty')
    ap.add_argument('--acceptance', choices=['armijo', 'nonmonotone', 'filter'], default='armijo')
    ap.add_argument('--lq-solver', choices=['serial', 'parallel', 'stagedense'], default='serial')
    ap.add_argument('--linear-rollout', action='store_true')
    ap.add_argument('--alpha-min', type=float, default=None, help='Line-search floor (solver default if unset)')
    ap.add_argument('--smooth', type=float, default=.005, help='Conservative fence-certificate smoothing temperature')
    ap.add_argument('--seed-lift', type=float, default=.2)
    ap.add_argument('--seed', choices=['benchmark', 'interp'], default='benchmark',
                    help='benchmark: analytic phase-timed seed + flight lift; interp: full-state linear '
                         'interpolation between the two rest boundaries, zero torque/force/acceleration')
    ap.add_argument('--control-reg', type=float, default=0., help='Fixed control residual coefficient (0 = pure feasibility)')
    ap.add_argument('--softplus', type=float, default=0., help='Smooth softplus envelope width for inequality rows (0 = exact positive part)')
    ap.add_argument('--w-geometry', type=float, default=1.)
    ap.add_argument('--w-force', type=float, default=1.)
    ap.add_argument('--w-state', type=float, default=1.)
    ap.add_argument('--w-control', type=float, default=1.)
    ap.add_argument('--w-inverse', type=float, default=1.)
    ap.add_argument('--w-terminal', type=float, default=1.)
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    if min(args.dt, args.iterations, args.tol, args.reg_init, args.reg_min, args.mu_init,
           args.w_geometry, args.w_force, args.w_state, args.w_control, args.w_inverse, args.w_terminal) <= 0:
        ap.error('grid, limits, tolerances, regularization and weights must be positive')
    if args.control_reg < 0 or args.smooth < 0 or args.time_limit < 0 or args.softplus < 0:
        ap.error('control regularization, smoothing and time limit must be nonnegative')
    cpus = pin_physical_cores(args.cores, args.core_offset)
    os.environ['OMP_NUM_THREADS'] = str(args.cores)

    import aligator as al
    import casadi as ca
    import numpy as np
    from common.spec_fence import load_fence_spec
    from common.plan import fence_plan
    from acados_bench import builders
    from acados_bench.fence_lifted import build_lifted
    from aligator_bench.orientation import matrix_orientation
    from aligator_bench.solve_fence_lifted import generate
    from aligator_bench.verify_fence_accuracy import verify as verify_zoh
    from aligator_bench.verify_fence_foh import verify as verify_foh
    from aligator_bench.foh_fence import build_foh_model, foh_seed
    from aligator_bench._foh_inverse import FOHInverse
    from aligator_bench._generated_sparse import GeneratedResidual
    from aligator_bench._inverse_rk4 import InverseRK4
    from aligator_bench import _uniform_run

    started = time.perf_counter()
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(rk4=True, cold_start=True, single_call=True, plastic_impact=False,
                    instant_liftoff=False, native_inverse=True, matrix_orientation=True,
                    compact_cost=True, lifted=True, seed=args.seed, hold=args.hold)
    record = dict(settings=settings, cpus=cpus, threads=args.cores,
                  solver='aligator.SolverFDDP' if args.solver == 'fddp' else 'aligator.SolverProxDDP',
                  aligator_version=al.__version__, solve_calls=1,
                  initialization=('analytic benchmark seed plus fixed 0.2 m flight lift; zero duals' if args.seed == 'benchmark'
                                  else 'full-state linear interpolation between the rest boundaries, zero torque/force/acceleration; zero duals'),
                  command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    args.out.with_suffix('.config.json').write_text(json.dumps(record, indent=2) + '\n')

    # ---- problem statement (shared with acados) and lifted RK4 model
    spec = load_fence_spec(ROOT / f'showcase_v2/fence/problems/{args.robot}_jump_fence.yaml')
    pl = fence_plan(spec, dt_target=args.dt, impulse_free=True)
    r = spec.robot
    nv = r.nv
    built = builders.build(pl, ROOT / 'comparisons/.aligator_generated' / 'schedule_only')
    if args.hold == 'zoh':
        directory = ROOT / 'comparisons/.aligator_generated' / \
            f'{args.robot}_{args.smooth}_matrixrot_nativeinv_compact_r{args.control_reg}'
        cfg = SimpleNamespace(smooth=args.smooth, rk4=True, plastic_impact=False, dyn_scale=1.,
                              hard_endpoints=False, hard_joints=False, exact_hessian=False,
                              condensing=None, lm=1e-3, native_max_iter=1, huber=0., planar_reduction=False)
        ocp = build_lifted(built, directory, cfg)
        model = ocp.model
        nreg = int(model.u.numel())
        matrix_orientation(model, built)
        # Native Pinocchio RNEA supplies the inverse-dynamics rows; drop the symbolic copy.
        model.cost_y_expr[built.bound_dim:built.bound_dim + built.inverse_dim] = 0
        first = built.bound_dim + built.inverse_dim
        model.cost_y_expr[first:first + nreg] *= args.control_reg / 1e-6
        names_running, names_terminal = row_names(built, model, nreg)
        stage_params = built.lifted_params
        inverse_check = built.inverse_check
    else:
        directory = ROOT / 'comparisons/.aligator_generated' / (f'{args.robot}_{args.smooth}_foh_v1_r{args.control_reg}' + (f'_sp{args.softplus}' if args.softplus > 0 else ''))
        model = build_foh_model(built, args.smooth, args.control_reg, args.softplus)
        names_running, names_terminal = list(model.names), list(model.names_e)
        stage_params = model.params
        inverse_check = None
    keep, keep_e = [], []
    for attr, names, out in (('cost_y_expr', names_running, keep), ('cost_y_expr_e', names_terminal, keep_e)):
        expression = getattr(model, attr)
        out.extend(i for i in range(expression.numel()) if not expression[i].is_zero())
        setattr(model, attr, expression[out])
    names_running = [names_running[i] for i in keep]
    names_terminal = [names_terminal[i] for i in keep_e]
    weight_of = dict(geometry=args.w_geometry, force=args.w_force, state=args.w_state,
                     control=args.w_control, inv=args.w_inverse, reg=1.)
    w_running = np.array([weight_of[family(n)] for n in names_running])
    w_terminal = args.w_terminal * np.array([weight_of[family(n)] for n in names_terminal])
    record['rows'] = dict(running=len(names_running), terminal=len(names_terminal), inverse=4 * nv,
                          families={f: int(sum(family(n) == f for n in names_running))
                                    for f in ('geometry', 'force', 'state', 'control', 'reg')})
    print(f'{args.robot}: generating residual C and exact first derivatives ({record["rows"]})', flush=True)
    library, funcs, digest = generate(model, directory, optimized=True)
    nx, nu = int(model.x.numel()), int(model.u.numel())
    space = al.manifolds.VectorSpace(nx)
    if args.hold == 'zoh':
        linear = ca.Function('linear', [model.p], [ca.jacobian(model.disc_dyn_expr, model.x),
                                                   ca.jacobian(model.disc_dyn_expr, model.u)])
    act = np.asarray(r.act_idx, dtype=np.int32)
    frames = np.array([c.frame_id for c in r.contacts], dtype=np.int32)
    dims = np.array([c.dim for c in r.contacts], dtype=np.int32)

    stages, inverse_fns = [], []
    for k in range(pl.nsteps):
        p = stage_params[k]
        fn = GeneratedResidual(nx, nu, len(names_running), str(library), 'running', p)
        if args.hold == 'zoh':
            inv = InverseRK4(nx, nu, r.model, act, frames, dims, pl.dts[k])
        else:
            inv = FOHInverse(r.model, act, frames, dims, pl.dts[k])
        cost = al.CostStack(space, nu)
        cost.addCost(al.QuadraticResidualCost(space, fn, np.diag(w_running ** 2)), 1.)
        cost.addCost(al.QuadraticResidualCost(space, inv, args.w_inverse ** 2 * np.eye(inv.nr)), 1.)
        A, B = linear(p) if args.hold == 'zoh' else model.dynamics(pl.dts[k])
        dyn = al.dynamics.LinearDiscreteDynamics(np.asarray(A), np.asarray(B), np.zeros(nx))
        stages.append(al.StageModel(cost, dyn))
        inverse_fns.append(inv)
    fn_e = GeneratedResidual(nx, nu, len(names_terminal), str(library), 'terminal', stage_params[-1])
    terminal = al.QuadraticResidualCost(space, fn_e, np.diag(w_terminal ** 2))
    problem = al.TrajOptProblem(pl.x0, stages, terminal)

    # ---- the analytic cold start (identical to acados_bench/solve_fence_native.py)
    xs = np.array(pl.xs_init, copy=True)
    phase = np.clip((pl.ts - spec.t_stance) / spec.t_air, 0, 1)
    xs[:, 2] += args.seed_lift * np.sin(np.pi * phase) ** 2
    xs[:, r.nq + 2] += args.seed_lift * np.pi / spec.t_air * np.sin(2 * np.pi * phase)
    acc = np.tile(np.diff(xs[:, r.nq:], axis=0) / np.asarray(pl.dts)[:, None], (1, 4))
    if args.hold == 'zoh':
        us = []
        for k in range(pl.nsteps):
            b = built.stage_bounds[k]
            us.append(np.concatenate([acc[k], np.clip(built.us_init_full[k], b['lbu'], b['ubu'])]))
        us = np.asarray(us)
    else:
        us = foh_seed(built, model, xs, acc)
    if args.seed == 'interp':
        # Noninformative cold start: the state interpolated linearly in time between the two
        # prescribed rest states (so v == 0 throughout), every control zero.
        frac = (np.asarray(pl.ts) / pl.ts[-1])[:, None]
        xs = (1 - frac) * pl.x0[None, :] + frac * pl.x_goal[None, :]
        us = np.zeros_like(us)
    record['seed_sha256'] = hashlib.sha256(xs.tobytes() + us.tobytes()).hexdigest()
    np.savez_compressed(args.out.with_suffix('.initial.npz'), xs=xs, us=us, ts=pl.ts)

    # ---- preflight: generated values/Jacobians vs CasADi and central differences
    rng = np.random.default_rng(21)
    worst = 0.
    for k in [0, pl.nsteps // 2, pl.nsteps - 1]:
        p = stage_params[k]
        fun = GeneratedResidual(nx, nu, len(names_running), str(library), 'running', p)
        d = fun.createData()
        x = xs[k] + rng.normal(size=nx) * 1e-4
        u = us[k] + rng.normal(size=nu) * 1e-4
        fun.evaluate(x, u, d); fun.computeJacobians(x, u, d)
        expected = np.asarray(funcs[0](x, u, p)).ravel()
        jx, ju = funcs[1](x, u, p)
        assert np.allclose(d.value, expected, rtol=1e-10, atol=1e-9)
        assert np.allclose(d.Jx, np.asarray(jx), rtol=1e-10, atol=1e-9)
        assert np.allclose(d.Ju, np.asarray(ju), rtol=1e-10, atol=1e-9)
        direction = rng.normal(size=nx + nu); direction /= np.linalg.norm(direction)
        h = 1e-6
        fd = (np.asarray(funcs[0](x + h * direction[:nx], u + h * direction[nx:], p)).ravel()
              - np.asarray(funcs[0](x - h * direction[:nx], u - h * direction[nx:], p)).ravel()) / (2 * h)
        ana = d.Jx @ direction[:nx] + d.Ju @ direction[nx:]
        err = float(np.max(np.abs(fd - ana)) / (1 + np.max(np.abs(ana))))
        worst = max(worst, err)
        assert err < 1e-5, err
        inv = inverse_fns[k]; di = inv.createData()
        inv.evaluate(x, u, di); inv.computeJacobians(x, u, di)
        if inverse_check is not None:
            expected = np.asarray(inverse_check(x, u, pl.dts[k])).ravel()
            np.testing.assert_allclose(di.value, expected, rtol=1e-10, atol=1e-8)
        else:
            expected = foh_inverse_numeric(r, act, x, u, pl.dts[k])
            np.testing.assert_allclose(di.value, expected, rtol=1e-9, atol=1e-7)
        J = np.c_[di.Jx.copy(), di.Ju.copy()]
        inv.evaluate(x + h * direction[:nx], u + h * direction[nx:], di); plus = di.value.copy()
        inv.evaluate(x - h * direction[:nx], u - h * direction[nx:], di); minus = di.value.copy()
        err = float(np.max(np.abs((plus - minus) / (2 * h) - J @ direction)) / (1 + np.max(np.abs(J @ direction))))
        worst = max(worst, err)
        assert err < 1e-5, err
    assert problem.checkIntegrity()

    # ---- the sole native solver call
    verbose = al.QUIET if args.quiet else al.VERBOSE
    if args.solver == 'fddp':
        solver = al.SolverFDDP(args.tol, verbose=verbose, reg_init=args.reg_init, max_iters=args.iterations)
        solver.force_initial_condition = True
        runner = _uniform_run.run_fddp
    else:
        strategy = {'armijo': al.SA_LINESEARCH_ARMIJO, 'nonmonotone': al.SA_LINESEARCH_NONMONOTONE,
                    'filter': al.SA_FILTER}[args.acceptance]
        solver = al.SolverProxDDP(args.tol, mu_init=args.mu_init, max_iters=args.iterations,
                                  verbose=verbose, sa_strategy=strategy)
        solver.max_al_iters = 1000
        solver.bcl_params.mu_lower_bound = 1e-14
        if args.linear_rollout:
            solver.rollout_type = al.ROLLOUT_LINEAR
        solver.linear_solver_choice = getattr(al, 'LQ_SOLVER_' + args.lq_solver.upper())
        runner = _uniform_run.run_proxddp
    if args.alpha_min is not None:
        solver.ls_params.alpha_min = args.alpha_min
    solver.reg_min = args.reg_min
    solver.reg_max = args.reg_max
    solver.setNumThreads(args.cores)
    solver.setup(problem)
    history = al.HistoryCallback(solver, False, True)
    solver.registerCallback('history', history)
    record.update(build_s=time.perf_counter() - started, generated_sha256=digest,
                  derivative_relative_error=worst, n_stages=pl.nsteps, nx=nx, nu=nu,
                  max_step_s=float(max(pl.dts)))
    print(f'{args.robot}: entering sole {record["solver"]}.run call '
          f'(cap {args.time_limit:.0f} s, {args.iterations} iterations, tol {args.tol:g})', flush=True)
    start = time.perf_counter()
    try:
        conv, timed_out, seen = runner(solver, problem, list(xs), list(us), args.time_limit)
    except Exception as exc:
        record.update(error=str(exc), solve_s=time.perf_counter() - start, passed=False)
        args.out.write_text(json.dumps(record, indent=2) + '\n')
        raise
    solve_s = time.perf_counter() - start
    costs = [float(v) for v in history.values]
    record.update(solve_s=solve_s, converged=bool(conv), timed_out=bool(timed_out),
                  iterations=int(solver.results.num_iters),
                  seconds_per_iteration=solve_s / max(1, int(solver.results.num_iters)),
                  penalty_cost=float(solver.results.traj_cost),
                  primal_infeasibility=float(solver.results.primal_infeas),
                  dual_infeasibility=float(solver.results.dual_infeas),
                  cost_history=costs[::max(1, len(costs) // 200)] + costs[-1:])
    print(f'{args.robot}: returned converged={conv} timed_out={timed_out} iterations={record["iterations"]} '
          f'solve_s={solve_s:.1f} cost={record["penalty_cost"]:.3e}', flush=True)
    X = np.array(list(solver.results.xs)); controls = np.array(list(solver.results.us))
    # Where does the remaining least-squares residual sit? (diagnostic only)
    worst_rows = {}
    for k in range(pl.nsteps):
        values = np.asarray(funcs[0](X[k], controls[k], stage_params[k])).ravel()
        for name, value in zip(names_running, np.abs(values)):
            key = family(name) if not name.startswith(('pos_', 'vel_', 'keepout_', 'selfcol_', 'mon_')) else name.split('_')[0]
            worst_rows[key] = max(worst_rows.get(key, 0.), float(value))
        di = inverse_fns[k].createData(); inverse_fns[k].evaluate(X[k], controls[k], di)
        worst_rows['inv'] = max(worst_rows.get('inv', 0.), float(np.max(np.abs(di.value))))
    values = np.asarray(funcs[2](X[-1], controls[-1], stage_params[-1])).ravel()
    for name, value in zip(names_terminal, np.abs(values)):
        worst_rows['terminal_' + family(name)] = max(worst_rows.get('terminal_' + family(name), 0.), float(value))
    record['residual_by_family'] = worst_rows
    print(f'{args.robot}: residual by family ' + ', '.join(f'{k}={v:.2e}' for k, v in sorted(worst_rows.items())), flush=True)
    acc = controls[:, :4 * nv]; U = controls[:, 4 * nv:]  # zoh: [tau, lam]; foh: [tau0, tau1, F0, F1]
    np.savez_compressed(args.out.with_suffix('.npz'), xs=X, us_full=U, acc=acc, ts=pl.ts)
    record['archive_sha256'] = hashlib.sha256(args.out.with_suffix('.npz').read_bytes()).hexdigest()
    record['passed'] = False
    args.out.write_text(json.dumps(record, indent=2) + '\n')
    try:
        if not np.isfinite(X).all() or not np.isfinite(controls).all():
            raise ValueError('Aligator returned a nonfinite trajectory')
        judged = (verify_zoh if args.hold == 'zoh' else verify_foh)(args.out)
    except Exception as exc:
        record.update(error=f'Independent verification failed: {type(exc).__name__}: {exc}',
                      process_wall_s=time.perf_counter() - started)
        args.out.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record, indent=2), flush=True)
        return 2
    record.update(independent=judged, passed=judged['passed'], violations_fine=judged['violations_fine'],
                  process_wall_s=time.perf_counter() - started)
    args.out.write_text(json.dumps(record, indent=2) + '\n')
    v = judged['violations_fine']
    worst_family = max((k for k in v if k != 'overall'), key=v.get)
    print(f'{args.robot}: {"PASS" if judged["passed"] else "FAIL"} max violation {v["overall"]:.3e} ({worst_family}); '
          f'solve {solve_s:.1f} s, {record["iterations"]} iterations', flush=True)
    return 0 if judged['passed'] else 2


if __name__ == '__main__':
    sys.exit(main())

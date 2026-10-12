"""Stock-Crocoddyl statement of the LEAP uneven-terrain restoration problem.

Software stack (the one of the fence progress experiment, comparisons/crocoddyl_impact_cone):
the installed unmodified Crocoddyl 3.2.1 provides SolverFDDP, ShootingProblem,
StateMultibody, ActionModelLQR (exact polynomial propagation of the lifted
transcription), ActionModelImpulseFwdDynamics + ImpulseModel3D/6D (the plastic
touchdown), CostModelSum, CostModelResidual, ActivationModelWeightedQuad,
ResidualModelState / ResidualModelControl. The physical residual evaluators are the
public models of the aligator terrain comparison (CasADi-generated bounded
rows, Pinocchio RNEA collocation inverse dynamics, classical contact acceleration,
coal capsule self-collision distance), bridged into Crocoddyl through the public
ResidualModelAbstract / ActionModelAbstract adapters of crocoddyl_impact_cone/native.cpp.
No optimizer code is copied, patched or reimplemented: the only solve call is stock FDDP.

The physical problem: showcase_v2/unified/README.md sections 1-5 (the restoration
twin, physical objective identically zero) on the trackable terrain set; every row,
event and bound comes from the robot yaml / problem yaml through tw_inputs /
tw_residuals; the cold start is the archived LEAP seed.
"""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / 'comparisons'), str(ROOT)]
for folder in ['crocoddyl_impact_cone', 'aligator_unified_fence/scripts', 'aligator_unified_terrain/scripts']:
    sys.path.append(str(ROOT / 'comparisons' / folder))

import crocoddyl as croc
import pinocchio as pin
import tw_residuals as residuals
import tw_verify as verification
from tw_inputs import ROBOTS, ALL_ROBOTS, geometry
from _public_models import MappedState, CollocationInverse, Geometry
from _terrain_models import QuadraticInverse
from _terrain_acceleration import ContactAcceleration
from aligator_bench._generated_sparse import GeneratedResidual
from _impact_native import ResidualBridge, CombinedAction

STOCK_COMPONENTS = ['SolverFDDP', 'ShootingProblem', 'StateMultibody', 'ActionModelLQR',
                    'ActionModelImpulseFwdDynamics', 'ImpulseModelMultiple', 'ImpulseModel3D', 'ImpulseModel6D',
                    'CostModelSum', 'CostModelResidual', 'ActivationModelWeightedQuad',
                    'ResidualModelState', 'ResidualModelControl']
PUBLIC_MODEL_ADAPTERS = {
    'crocoddyl_impact_cone/native.cpp': 'ResidualBridge (StageFunction -> ResidualModelAbstract), CombinedAction (stock dynamics + stock costs -> ActionModelAbstract), run (stock solve with a Python callback)',
    'aligator_unified_fence/scripts/public_models.cpp': 'MappedState (exact dense-state pull-back), CollocationInverse (linear hold RNEA), Geometry (coal capsule self-distance, clipped)',
    'aligator_unified_terrain/scripts/terrain_models.cpp': 'QuadraticInverse (quadratic-hold RNEA at s = 0, 1/2, 1)',
    'aligator_unified_terrain/scripts/terrain_acceleration.cpp': 'ContactAcceleration (classical frame acceleration, LOCAL_WORLD_ALIGNED)',
    'aligator_bench/generated_sparse.cpp': 'GeneratedResidual (CasADi-generated bounded rows)',
}


def lqr(A, B):
    """Stock linear action x+ = A x + B u with zero cost."""
    nx, nu = B.shape
    return croc.ActionModelLQR(A, B, np.zeros((nx, nx)), np.zeros((nu, nu)), np.zeros((nx, nu)),
                               np.zeros(nx), np.zeros(nx), np.zeros(nu))


def stock_impact(state, robot):
    """Zero-restitution all-contact plastic impulse (every terrain support mode has all
    contacts active); JMinvJt damping 1e-9 as in LEAP's one-shot proximal reset."""
    impulses = croc.ImpulseModelMultiple(state)
    for c in robot.contacts:
        cls = croc.ImpulseModel3D if c.dim == 3 else croc.ImpulseModel6D
        impulses.addImpulse(c.name, cls(state, c.frame_id, pin.LOCAL_WORLD_ALIGNED))
    return croc.ActionModelImpulseFwdDynamics(state, impulses, croc.CostModelSum(state, 0), 0., 1e-9, True)


def build(robot, dt=.01, hold='quadratic'):
    if robot not in ALL_ROBOTS:
        raise ValueError(f'Robot must be one of {ALL_ROBOTS}')
    if not np.isfinite(dt) or dt <= 0 or dt > .01:
        raise ValueError('dt must be positive and no greater than 0.01 seconds')
    info, mesh, xs, us, fields = residuals.cold(robot, dt, hold)
    r, s = info.r, info.spec
    nv, nt, nf, nu = residuals.dimensions(r)
    nx = 2 * nv
    state = croc.StateMultibody(r.model)
    sym = residuals.symbolic(info)
    geom, grows = geometry(info)
    margins = np.array([v for _, v in grows])
    frames = np.array([c.frame_id for c in r.contacts], np.int32)
    dims = np.array([c.dim for c in r.contacts], np.int32)
    actions, checks = [], []
    rows_total = 0

    def add(costs, name, fn, weight=1.):
        # 1/2 ||fn||^2 with the Gauss-Newton Hessian, the stock quadratic activation
        costs.addCost(name, croc.CostModelResidual(state, ResidualBridge(state, fn)), weight)

    for step in mesh:
        h, kind, mode = step['h'], step['kind'], step['mode']
        costs = croc.CostModelSum(state, nu)
        A, B = np.eye(nx), np.zeros((nx, nu))
        check = dict(kind=kind, rows=0, bounds=[])
        if kind == 'initial':
            # Free initial configuration: x1 = (u[:nv], 0) from a fixed dummy zero state;
            # every other dummy control is driven to zero.
            A[:] = 0.; B[:nv, :nv] = np.eye(nv)
            w = np.ones(nu); w[:nv] = 0.
            costs.addCost('initial_dummy_controls', croc.CostModelResidual(
                state, croc.ActivationModelWeightedQuad(w), croc.ResidualModelControl(state, nu)), 1.)
            check['rows'] = nu - nv
            actions.append(CombinedAction(lqr(A, B), costs)); checks.append(check); rows_total += check['rows']
            continue
        active = np.array([c.name in s.modes[mode].active for c in r.contacts], np.int32)
        for j, fraction in enumerate([0., .5, 1.] if kind == 'flow' else [0.]):
            p = residuals.bounds(info, step, sym, fraction)
            fn = GeneratedResidual(nx, nu, sym.nr, str(sym.library), 'running', p)
            Aa, Bb = np.eye(nx), np.zeros((nx, nu)); qw, vw, _ = verification.weights(fraction)
            Aa[:nv, nv:] = h * fraction * np.eye(nv)
            for a in range(3):
                Bb[:nv, a * nv:(a + 1) * nv] = h * h * qw[a] * np.eye(nv)
                Bb[nv:, a * nv:(a + 1) * nv] = h * vw[a] * np.eye(nv)
            functions = [(f'path_{j}', fn, residuals.posed_rows(p, sym.nr))]
            if len(grows):
                functions.append((f'self_collision_{j}', Geometry(nu, r.model, geom, margins), len(grows)))
            if kind == 'flow':
                functions.append((f'stance_acceleration_{j}', ContactAcceleration(nu, r.model, frames, dims, active, j),
                                  int(dims[active == 1].sum())))
            for name, residual, count in functions:
                add(costs, name, MappedState(residual, Aa, Bb) if fraction else residual)
                check['rows'] += count
            check['bounds'].append(p)
        if kind == 'flow':
            inverse = QuadraticInverse if hold == 'quadratic' else CollocationInverse
            inv = inverse(r.model, np.array(r.act_idx, np.int32), frames, dims, h)
            add(costs, 'inverse_dynamics', inv); check['rows'] += 3 * nv
            # exact endpoint propagation of the quartic q / cubic v
            A[:nv, nv:] = h * np.eye(nv)
            for j, value in enumerate([1, 2, 0]): B[:nv, j * nv:(j + 1) * nv] = value * h * h / 6 * np.eye(nv)
            for j, value in enumerate([1, 4, 1]): B[nv:, j * nv:(j + 1) * nv] = value * h / 6 * np.eye(nv)
        else:
            costs.addCost('event_dummy_controls', croc.CostModelResidual(
                state, croc.ResidualModelControl(state, np.zeros(nu))), 1.)
            check['rows'] += nu
        dynamics = stock_impact(state, r) if kind == 'impact' else lqr(A, B)
        actions.append(CombinedAction(dynamics, costs)); checks.append(check); rows_total += check['rows']
    # Terminal rest (the terminal placements and boxes are the last flow stage's s = 1 rows).
    terminal_costs = croc.CostModelSum(state, nu)
    terminal_costs.addCost('terminal_rest', croc.CostModelResidual(
        state, croc.ActivationModelWeightedQuad(np.r_[np.zeros(nv), np.ones(nv)]),
        croc.ResidualModelState(state, np.zeros(nx), nu)), 1.)
    terminal = CombinedAction(lqr(np.eye(nx), np.zeros((nx, nu))), terminal_costs)
    rows_total += nv
    problem = croc.ShootingProblem(xs[0], actions, terminal)
    return SimpleNamespace(family='terrain', robot=robot, info=info, mesh=mesh, xs=xs, us=us, fields=fields,
                           problem=problem, verification=verification, symbolic_digest=sym.digest,
                           generated_library=Path(sym.library), sym=sym, geometry='original_terrain_face_certificate',
                           self_collision='coal capsule distance (optimizer) / normalized squared gap (judge)',
                           hold=hold, checks=checks, state=state, posed_rows=rows_total,
                           dimensions=dict(nx=nx, nu=nu, nv=nv, nt=nt, nf=nf, count=3 if hold == 'quadratic' else 2),
                           stages=len(mesh))


def audit(built, xs, us, fine_dt=.01):
    return verification.audit(built.info, built.mesh, list(xs), list(us), fine_dt)

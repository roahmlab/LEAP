"""Impact-cone fence modeling on unmodified stock Crocoddyl.

The newer benchmark's polynomial transcription is retained. Existing ALIGATOR
StageFunction components are used only as residual evaluators, through a native
ResidualModelAbstract adapter. All costs, impulse actions, linear propagation,
shooting and optimization are Crocoddyl components.
"""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ROBOTS = ('hopper2d', 'go2', 'h1', 'g1_29dof', 'talos')
sys.path[:0] = [str(HERE), str(ROOT / 'comparisons'), str(ROOT)]
sys.path.append(str(ROOT / 'comparisons/aligator_unified_fence/scripts'))

import crocoddyl as croc
import pinocchio as pin
from inputs import inputs
import residuals as physical
import verify as verification
from _public_models import MappedState, CollocationInverse
from aligator_bench._generated_sparse import GeneratedResidual
from _impact_native import ResidualBridge, CombinedAction, Geometry, ImpulseCone


def geometry_arrays(info):
    r, s = info.r, info.spec
    names = list(r.capsules)
    caps = np.array([[r.frame_id(c.frame), r.frame_id(c.b_frame), c.radius,
                      *c.a_local, *c.b_local] for c in r.capsules.values()])
    boxes = np.array([np.r_[c, h] for c, h in s.obstacle_boxes]).reshape(-1, 6)
    pairs = np.array([[names.index(a), names.index(b), safe]
                      for a, b, safe in s.selfcol_pairs]).reshape(-1, 3)
    return caps, boxes, pairs


def stock_impact(state, robot):
    impulses = croc.ImpulseModelMultiple(state)
    for c in robot.contacts:
        cls = croc.ImpulseModel3D if c.dim == 3 else croc.ImpulseModel6D
        impulses.addImpulse(c.name, cls(state, c.frame_id, pin.LOCAL_WORLD_ALIGNED))
    return croc.ActionModelImpulseFwdDynamics(
        state, impulses, croc.CostModelSum(state, 0), 0., 1e-9, True)


def build(robot, dt=.01):
    if robot not in ROBOTS:
        raise ValueError(f'Robot must be one of {ROBOTS}')
    if not np.isfinite(dt) or dt <= 0 or dt > .01:
        raise ValueError('dt must be positive and no greater than 0.01 seconds')
    info, mesh, xs, us, fields = physical.cold(robot, dt)
    r, s = info.r, info.spec
    nv, nt, nf, nu = physical.dimensions(r)
    nx = 2 * nv
    state = croc.StateMultibody(r.model)
    sym = physical.symbolic(info)
    geometry = Geometry(nu, r.model, *geometry_arrays(info), s.obstacle_margin)
    cone = ImpulseCone(nu, r.model, list(info.cms[1]), s.mu, s.friceps)
    actions = []
    checks = []

    def add(costs, name, function, inequality=False):
        residual = ResidualBridge(state, function)
        if inequality:
            bounds = croc.ActivationBounds(np.full(function.nr, -np.inf), np.zeros(function.nr))
            cost = croc.CostModelResidual(state, croc.ActivationModelQuadraticBarrier(bounds), residual)
        else:
            cost = croc.CostModelResidual(state, residual)
        costs.addCost(name, cost, 1.)

    for step in mesh:
        h, kind = step['h'], step['kind']
        costs = croc.CostModelSum(state, nu)
        p = physical.bounds(info, step, sym)
        fn = GeneratedResidual(nx, nu, sym.nr, str(sym.library), 'running', p)
        for j, fraction in enumerate([0., .5, 1.] if kind == 'flow' else [0.]):
            A, B = np.eye(nx), np.zeros((nx, nu))
            qw, vw, _ = verification.weights(fraction)
            A[:nv, nv:] = h * fraction * np.eye(nv)
            for i in range(3):
                B[:nv, i*nv:(i+1)*nv] = h*h*qw[i]*np.eye(nv)
                B[nv:, i*nv:(i+1)*nv] = h*vw[i]*np.eye(nv)
            add(costs, f'path_{j}', MappedState(fn, A, B) if fraction else fn)
            add(costs, f'geometry_{j}', MappedState(geometry, A, B) if fraction else geometry, True)
        A, B = np.eye(nx), np.zeros((nx, nu))
        if kind == 'flow':
            inv = CollocationInverse(r.model, np.array(r.act_idx, np.int32),
                                     np.array([c.frame_id for c in r.contacts], np.int32),
                                     np.array([c.dim for c in r.contacts], np.int32), h)
            add(costs, 'inverse_dynamics', inv)
            A[:nv, nv:] = h*np.eye(nv)
            for j, value in enumerate([1, 2, 0]):
                B[:nv, j*nv:(j+1)*nv] = value*h*h/6*np.eye(nv)
            for j, value in enumerate([1, 4, 1]):
                B[nv:, j*nv:(j+1)*nv] = value*h/6*np.eye(nv)
        else:
            costs.addCost('event_dummy_controls', croc.CostModelResidual(
                state, croc.ResidualModelControl(state, np.zeros(nu))), 1.)
        if kind == 'impact':
            add(costs, 'impact_cone', cone, True)
            dynamics = stock_impact(state, r)
        else:
            dynamics = croc.ActionModelLQR(A, B, np.zeros((nx, nx)),
                                           np.zeros((nu, nu)), np.zeros((nx, nu)), np.zeros(nx), np.zeros(nx), np.zeros(nu))
        actions.append(CombinedAction(dynamics, costs))
        checks.append(dict(dynamics=dynamics, costs=costs, bounds=p))

    # Terminal feasibility includes the actual endpoint's path and geometry rows.
    terminal_costs = croc.CostModelSum(state, nu)
    goal = np.r_[s.q_hold2, np.zeros(nv)]
    terminal_costs.addCost('terminal_state', croc.CostModelResidual(
        state, croc.ResidualModelState(state, goal, nu)), 1.)
    terminal_step = dict(mode=2, kind='terminal', t=info.seed.sol.duration, h=0.)
    p = physical.bounds(info, terminal_step, sym)
    add(terminal_costs, 'terminal_path', GeneratedResidual(nx, nu, sym.nr, str(sym.library), 'running', p))
    add(terminal_costs, 'terminal_geometry', geometry, True)
    terminal_dyn = croc.ActionModelLQR(np.eye(nx), np.zeros((nx, nu)), np.zeros((nx, nx)),
                                     np.zeros((nu, nu)), np.zeros((nx, nu)), np.zeros(nx), np.zeros(nx), np.zeros(nu))
    terminal = CombinedAction(terminal_dyn, terminal_costs)
    problem = croc.ShootingProblem(xs[0], actions, terminal)
    return SimpleNamespace(family='fence', info=info, mesh=mesh, xs=xs, us=us,
                           fields=fields, problem=problem, verification=verification,
                           symbolic_digest=sym.digest, generated_library=Path(sym.library), geometry='leap_rounded', hold='linear',
                           checks=checks, cone=cone, geometry_residual=geometry, state=state)


def audit(built, xs, us, fine_dt=.01):
    return verification.audit(built.info, built.mesh, list(xs), list(us), fine_dt)

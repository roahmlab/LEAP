"""Crocoddyl and Aligator runs as tracking references, tracked by the same code path
as LEAP.

A stored solution of a run (its final iterate, or any candidate of the compute-budget
study) is turned into a reference object with the interface `rollout.run(reference=...)`
and `evaluate.evaluate` read for a LEAP archive (`spec`, `robot`, `plan`,
`eval(t) -> RefSample`, `q0`, `duration`, `touchdown_events`, `foot_phase_at`,
`impact_times`, `scene_boxes`, `planarlock`, `settings`, `npz_path`, ...).
`track_recorded` then runs `rollout.run` / `evaluate.evaluate_dir` and writes the files
`rollout.track_archive` writes for LEAP, with `method` = crocoddyl / aligator.

Inputs of a run:
  * the problem yaml is the copy stored with the run (Crocoddyl `<run>/inputs/<repo
    path>`, Aligator `<run>/reference_inputs/`, listed in `protocol.json` `input_sha256` /
    `config.json` `inputs`), written to `work_dir` with its `robot:` line resolved against
    `new_src/robots/` (`reference.materialize_problem_yaml`);
  * the robot yaml and URDF are `new_src/robots/<robot>.yaml` and the URDF it names in
    `new_src/assets/`;
  * the run folder is the one `new_src/baselines/<method>/run.sh` wrote (`--run-dir`).

What comes from where for a recorded candidate:
  * q, v, a (quartic / cubic / quadratic per flow stage from its three acceleration
    samples), torque and contact wrench (the stage's linear or quadratic Bernstein
    hold), velocity jumps at the plastic-impact stages (right limit at every event),
    the terminal state xs[-1] ............................. the recorded primal (mesh, xs, us)
  * scene geometry, friction, CoP box, planar lock, terminal footholds and rest pose,
    the contact schedule and the knot grid (`fence_plan` at dt 0.01 for Crocoddyl and
    0.005 for Aligator; `terrain_plan` at dt 0.01) .... the problem yaml
Past the final time the reference holds xs[-1] at rest with the end torque and wrench
of the last flow stage.

Candidate ids: Crocoddyl the record index ('60'; on the terrain the `progress.csv`
index, mapped to its stored candidate), Aligator 'phase:call:sample' ('3:32:152');
'best' = the run's own selection rule. `new_src/baselines/<method>/README.md` shows how to
print a run's final iterate id.
"""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import yaml

from .paths import robot_yaml_path
from .plan import fence_plan, terrain_plan
from .reference import YAML_SUFFIX, FenceReference, RefSample, TerrainReference, materialize_problem_yaml
from .spec_fence import load_fence_spec
from .spec_terrain import load_terrain_spec

METHODS = ("crocoddyl", "aligator")

SCHEMA = "crocoddyl-terrain-progress-v1"
FINE_DT = 0.01
EVENT_TOL = 1e-9      # right-limit lookup tolerance at mesh events (roundoff of the cumulative stage times)
NPZ = "best_observed_solution.npz"
# Simulation robot -> the Aligator terrain experiment's run names for it, preferred first
# (`g1_29dof_repaired` solves the g1 problem of problems/g1_29dof_terrain_walk.yaml).
RUN_NAMES = {"g1_29dof": ("g1_29dof_repaired", "g1_29dof")}


# ===================================================================== inputs
def _stored_inputs(entries: dict, folder: Path, layout: str) -> dict:
    """{recorded input name: the run's stored copy}. `layout`: 'tree' (Crocoddyl:
    `inputs/<repo path>`), 'flat' (Aligator terrain: `reference_inputs/<basename>`) or
    'numbered' (Aligator fence: `reference_inputs/<k>_<basename>`, k = the record order)."""
    out = {}
    for k, name in enumerate(entries):
        if layout == "tree":
            cands = [folder / name]
        else:
            base = Path(name).name
            cands = ([folder / f"{k}_{base}"] if layout == "numbered" else []) + [folder / base]
        path = next((c for c in cands if c.exists()), cands[0])
        if not path.exists():
            raise FileNotFoundError(f"Archived input missing: {path}")
        out[name] = path
    return out


def _robot_files(problem_text: str) -> dict:
    """new_src's robot yaml (the problem's `robot:` by basename) and the URDF it names."""
    doc = yaml.safe_load(problem_text)
    ryaml = robot_yaml_path(doc["robot"])
    urdf = (ryaml.parent / yaml.safe_load(ryaml.read_text())["urdf"]).resolve()
    return dict(robot_yaml=str(ryaml), urdf=str(urdf))


def _problem_yaml(task: str, robot: str, stored: dict, name: str, work_dir):
    """(the yaml the spec loaders read, its stored source, the robot files)."""
    src = stored[name]
    text = src.read_text()
    robot_files = _robot_files(text)
    wd = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="leaptrack_"))
    dest = materialize_problem_yaml(text, wd / f"{robot}_{YAML_SUFFIX[task]}.yaml")
    return dest, src, robot_files


def _run_dir_of(run_dir, robot: str, marker: str) -> Path:
    """The run folder itself, or its parent."""
    d = Path(run_dir)
    if (d / marker).exists():
        return d
    if (d / robot / marker).exists():
        return d / robot
    raise FileNotFoundError(f"no {marker} in {d} or {d / robot}")


# ============================================================ the references
def dense_fence(robot, x, u, h, s):
    """The archive's quadratic acceleration integrated exactly (verify.dense)."""
    nv, nt = robot.nv, robot.nu
    nf = sum(c.dim for c in robot.contacts)
    aw = np.array([2*s*s-3*s+1, 4*s-4*s*s, 2*s*s-s])
    vw = np.array([2*s**3/3-1.5*s*s+s, 2*s*s-4*s**3/3, 2*s**3/3-.5*s*s])
    qw = np.array([s**4/6-.5*s**3+.5*s*s, 2*s**3/3-s**4/3, s**4/6-s**3/6])
    rates = u[:3*nv].reshape(3, nv)
    q = x[:nv] + h*s*x[nv:] + h*h*(qw @ rates)
    v = x[nv:] + h*(vw @ rates)
    tau = (1-s)*u[3*nv:3*nv+nt] + s*u[3*nv+nt:3*nv+2*nt]
    force = (1-s)*u[3*nv+2*nt:3*nv+2*nt+nf] + s*u[-nf:]
    return q, v, aw @ rates, tau, force


def weights(s):
    """Quartic q / cubic v / quadratic a weights of the three acceleration samples
    (copy of tw_verify.weights)."""
    a = np.array([2 * s * s - 3 * s + 1, 4 * s - 4 * s * s, 2 * s * s - s])
    v = np.array([2 * s ** 3 / 3 - 1.5 * s * s + s, 2 * s * s - 4 * s ** 3 / 3, 2 * s ** 3 / 3 - .5 * s * s])
    q = np.array([s ** 4 / 6 - .5 * s ** 3 + .5 * s * s, 2 * s ** 3 / 3 - s ** 4 / 3, s ** 4 / 6 - s ** 3 / 6])
    return q, v, a


def dense(robot, x, u, h, s):
    """q, v, a, torque, wrench of a flow stage at fraction s of its duration h
    (copy of tw_verify.dense)."""
    nv, nt = robot.nv, robot.nu
    nf = sum(c.dim for c in robot.contacts)
    count = (len(u) - 3 * nv) // (nt + nf)
    if count not in (2, 3) or len(u) != 3 * nv + count * (nt + nf):
        raise ValueError("Unknown control hold")
    rates = u[:3 * nv].reshape(3, nv)
    qw, vw, aw = weights(s)
    q = x[:nv] + h * s * x[nv:] + h * h * (qw @ rates)
    v = x[nv:] + h * (vw @ rates)
    tau = u[3 * nv:3 * nv + count * nt].reshape(count, nt)
    force = u[3 * nv + count * nt:].reshape(count, nf)
    hold = np.array([(1 - s) ** 2, 2 * s * (1 - s), s * s]) if count == 3 else np.array([1 - s, s])
    return q, v, aw @ rates, hold @ tau, hold @ force


class RecordedFenceReference(FenceReference):
    @property
    def impact_times(self):
        return [step['t'] for step in self.mesh if step['kind'] == 'impact']

    def native_eval(self, t):
        # Right limit at knots and events, without smoothing velocity jumps.
        k = int(np.clip(np.searchsorted(self.flow_times, t, side='right')-1,
                        0, len(self.flow_indices)-1))
        i = self.flow_indices[k]
        step = self.mesh[i]
        s = float(np.clip((t-step['t'])/step['h'], 0., 1.))
        return dense_fence(self.robot, self.native_xs[i], self.native_us[i], step['h'], s)

    def _hold_controls(self):
        _, _, _, u, lam = self.native_eval(self.duration)
        return u.copy(), lam.copy()

    def eval(self, t):
        q, v, a, u, row = self.native_eval(t)
        if t >= self.duration:
            q, v, a = self.q_final(), np.zeros(self.nv), np.zeros(self.nv)
        phase, progress = self.phase_at(t)
        scheduled = self.scheduled_at(t)
        pt = self.phase_times
        return RefSample(t=float(t), q=q.copy(), v=v.copy(), a=a.copy(), u=u.copy(),
                         lam={c: row[sl].copy() for c, sl in self.lam_slices.items() if c in scheduled},
                         scheduled=scheduled, targets=self.targets_at(t), phase=phase,
                         phase_s=progress, stage=self.stage_at(t), foot_phase=self.foot_phase_at(t),
                         phase_duration=pt[phase+1]-pt[phase] if phase+1 < len(pt) else 0.)


class RecordedTerrainReference(TerrainReference):
    @property
    def impact_times(self):
        return [step["t"] for step in self.mesh if step["kind"] == "impact"]

    def native_eval(self, t):
        # Right limit at knots and events, without smoothing velocity jumps: the
        # lookup tolerates EVENT_TOL so a query at the event instant lands on the
        # post-impact stage.
        k = int(np.clip(np.searchsorted(self.flow_times, t + EVENT_TOL, side="right") - 1, 0, len(self.flow_indices) - 1))
        i = self.flow_indices[k]
        step = self.mesh[i]
        s = float(np.clip((t - step["t"]) / step["h"], 0.0, 1.0))
        return dense(self.robot, self.native_xs[i], self.native_us[i], step["h"], s)

    def _hold_controls(self):
        _, _, _, u, lam = self.native_eval(self.duration)
        return u.copy(), lam.copy()

    def eval(self, t):
        q, v, a, u, row = self.native_eval(t)
        if t >= self.duration:
            q, v, a = self.q_final(), np.zeros(self.nv), np.zeros(self.nv)
        phase, progress = self.phase_at(t)
        scheduled = self.scheduled_at(t)
        pt = self.phase_times
        return RefSample(t=float(t), q=q.copy(), v=v.copy(), a=a.copy(), u=u.copy(),
                         lam={c: row[sl].copy() for c, sl in self.lam_slices.items() if c in scheduled},
                         scheduled=scheduled, targets=self.targets_at(t), phase=phase,
                         phase_s=progress, stage=self.stage_at(t), foot_phase=self.foot_phase_at(t),
                         phase_duration=pt[phase + 1] - pt[phase] if phase + 1 < len(pt) else 0.0)


def _new_reference(cls, method, robot, path, json_path, spec, plan, settings):
    r = spec.robot
    ts = np.asarray(plan.ts)
    nf = sum(c.dim for c in r.contacts)
    return cls(method=method, robot_name=robot, npz_path=path, json_path=json_path, spec=spec, plan=plan, ts=ts,
               xs=np.zeros((len(ts), 2 * r.nv)), us=np.zeros((len(ts) - 1, r.nu)),
               lams=np.zeros((len(ts) - 1, nf)), a1=np.zeros((len(ts) - 1, r.nv)),
               a4=np.zeros((len(ts) - 1, r.nv)), settings=settings)


def _attach_native(ref, mesh, xs, us):
    ref.mesh, ref.native_xs, ref.native_us = mesh, xs, us
    ref.flow_indices = [i for i, step in enumerate(mesh) if step['kind'] == 'flow']
    ref.flow_times = np.array([mesh[i]['t'] for i in ref.flow_indices])


def _fill_grid(ref, xs):
    """The legacy knot-grid fields, sampled from the native curves (right limits)."""
    ts = ref.ts
    for k, t in enumerate(ts):
        q, v, a, u, lam = ref.native_eval(float(t))
        ref.xs[k] = np.r_[q, v]
        if k < len(ts)-1:
            ref.us[k], ref.lams[k], ref.a1[k] = u, lam, a
            ref.a4[k] = ref.native_eval(float(np.nextafter(ts[k+1], -np.inf)))[2]
    ref.xs[-1] = xs[-1]


# ----------------------------------------------------------- Crocoddyl fence
def load_crocoddyl_fence(robot, run_dir, index=None, work_dir=None):
    """`index=None`: the best committed snapshot (the study's selection rule);
    an integer: that committed snapshot, whatever its violation (budget rollouts)."""
    directory = _run_dir_of(run_dir, robot, 'protocol.json')
    protocol = json.loads((directory/'protocol.json').read_text())
    if protocol['schema'] != 'crocoddyl-impact-cone-v1':
        raise ValueError('Unexpected Crocoddyl archive schema')
    # Read the archived inputs.
    stored = _stored_inputs(protocol['input_sha256'], directory/'inputs', 'tree')
    yaml_name = next(n for n in protocol['input_sha256'] if n.endswith(f'{robot}_jump_fence.yaml'))
    yaml_path, source, robot_files = _problem_yaml('fence', robot, stored, yaml_name, work_dir)
    spec = load_fence_spec(yaml_path, allow_impact_cone=True)
    plan = fence_plan(spec, dt_target=.01, impulse_free=False)
    path = directory/'iterations.h5'
    with h5py.File(path, 'r') as f:
        count = int(f.attrs['committed_iterations'])
        records = [json.loads(r) for r in f['records_json'][:count]]
        if index is None:
            best = min(records, key=lambda r: (r['maximum_violation'], r['index']))
            selection = 'minimum recorded maximum_violation; earliest tie'
        else:
            best = records[int(index)]
            if int(best['index']) != int(index):
                raise ValueError(f'Record {index} is not at position {index}: {path}')
            selection = f'requested committed snapshot {int(index)}'
        index = int(best['index'])
        mesh = json.loads(f.attrs['mesh_json'])
        xs, us = f['xs'][index], f['us'][index]
        names = json.loads(f.attrs['joint_names'])
        if names != list(spec.robot.model.names)[1:]:
            raise ValueError('Archive and robot joint orders differ')
    ref = _new_reference(RecordedFenceReference, 'crocoddyl', robot, path, directory/'result.json', spec, plan,
                         dict(source=str(path), snapshot_index=index, selection=selection, selected_record=best,
                              problem_yaml=str(yaml_path), problem_yaml_source=str(source), robot_files=robot_files,
                              gate=best['maximum_violation'], dt=.01))
    _attach_native(ref, mesh, xs, us)
    if not np.isclose(mesh[-1]['t']+mesh[-1]['h'], ref.ts[-1], atol=1e-10):
        raise ValueError('Archive and problem durations differ')
    _fill_grid(ref, xs)
    return ref


# --------------------------------------------------------- Crocoddyl terrain
def is_recorded_terrain_archive(directory: Path) -> bool:
    p = Path(directory) / "protocol.json"
    if not p.exists() or not (Path(directory) / "iterations.h5").exists():
        return False
    try:
        return json.loads(p.read_text()).get("schema") == SCHEMA
    except ValueError:
        return False


def select_candidate(directory: Path) -> dict:
    """The stored candidate with the minimum offline-audited maximum_violation
    (earliest stored index on a tie), with its audit record."""
    audits = [json.loads(line) for line in (directory / "audits.jsonl").read_text().splitlines() if line.strip()]
    if not audits:
        raise ValueError(f"{directory}: no offline audits")
    audits = [a for a in audits if a.get("finite", True)]
    best = min(audits, key=lambda a: (a["maximum_violation"], a["stored_index"]))
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        sb = summary.get("best") or {}
        if sb and (int(sb["stored_index"]) != int(best["stored_index"]) or
                   not np.isclose(sb["maximum_violation"], best["maximum_violation"], rtol=0, atol=1e-15)):
            raise ValueError(f"{directory}: audits.jsonl best (stored {best['stored_index']}, "
                             f"{best['maximum_violation']:.3e}) != summary.json best "
                             f"(stored {sb['stored_index']}, {sb['maximum_violation']:.3e})")
    return best


def load_crocoddyl_terrain(robot, run_dir, index=None, work_dir=None):
    """`index=None`: the best offline-audited stored candidate (the study's selection
    rule); an integer: that STORED candidate (`stored_index`), whatever its violation."""
    directory = _run_dir_of(run_dir, robot, "protocol.json")
    protocol = json.loads((directory / "protocol.json").read_text())
    if protocol["schema"] != SCHEMA:
        raise ValueError("Unexpected Crocoddyl archive schema")
    if protocol["robot"] != robot:
        raise ValueError(f"Archive robot {protocol['robot']} != {robot}")
    # The archived inputs.
    stored = _stored_inputs(protocol["input_sha256"], directory / "inputs", "tree")
    yaml_name = next(n for n in protocol["input_sha256"] if "/problems/" in n and n.endswith(".yaml"))
    yaml_path, source, robot_files = _problem_yaml("terrain", robot, stored, yaml_name, work_dir)
    spec = load_terrain_spec(yaml_path)
    plan = terrain_plan(spec, dt_target=FINE_DT, kp=100.0, impulse_free=False)
    if index is None:
        best = select_candidate(directory)
        selection = "minimum offline-audited maximum_violation (audits.jsonl); earliest stored index on a tie"
    else:
        audits = [json.loads(line) for line in (directory / "audits.jsonl").read_text().splitlines() if line.strip()]
        found = [a for a in audits if int(a["stored_index"]) == int(index)]
        if len(found) != 1:
            raise ValueError(f"{directory}: stored candidate {index} has {len(found)} offline audits")
        best = found[0]
        selection = f"requested stored candidate {int(index)}"
    index = int(best["stored_index"])
    path = directory / "iterations.h5"
    with h5py.File(path, "r") as f:
        count = int(f.attrs["committed_iterations"])
        if not 0 <= index < count:
            raise ValueError(f"Selected candidate {index} not committed ({count})")
        mesh = json.loads(f.attrs["mesh_json"])
        xs, us = f["xs"][index], f["us"][index]
        names = json.loads(f.attrs["joint_names"])
        if names != list(spec.robot.model.names)[1:]:
            raise ValueError("Archive and robot joint orders differ")
        contacts = json.loads(f.attrs["contacts"])
        if [(c["name"], int(c["dim"])) for c in contacts] != [(c.name, int(c.dim)) for c in spec.robot.contacts]:
            raise ValueError("Archive and robot contacts differ")
        records = [json.loads(r) for r in f["records_json"][:int(f.attrs["committed_records"])]]
    record = next((r for r in records if r.get("stored_index") == index and r.get("trajectory_changed", True)), None)
    ts = np.asarray(plan.ts)
    ref = _new_reference(RecordedTerrainReference, "crocoddyl", robot, path, directory / "result.json", spec, plan,
                         dict(source=str(path), snapshot_index=index, stored_index=index,
                              record_index=(None if record is None else int(record["index"])),
                              solver_s=(None if record is None else float(record["solver_s"])),
                              wall_s=(None if record is None else float(record["wall_s"])),
                              selection=selection, selected_audit=best, selected_record=record,
                              problem_yaml=str(yaml_path), problem_yaml_source=str(source), robot_files=robot_files,
                              gate=float(best["maximum_violation"]), passed=bool(best["passed"]), dt=FINE_DT,
                              hold=protocol["hold"], schema=SCHEMA,
                              stop_reason=json.loads((directory / "status.json").read_text()).get("stop_reason")))
    _attach_native(ref, mesh, xs, us)
    if not np.isclose(mesh[-1]["t"] + mesh[-1]["h"], ts[-1], atol=1e-9):
        raise ValueError(f"Archive duration {mesh[-1]['t'] + mesh[-1]['h']} != problem {ts[-1]}")
    _fill_grid(ref, xs)
    return ref


# ------------------------------------------------------------ Aligator fence
def _phase_dir(directory, summary, phase):
    """`summary['phases'][phase]` was recorded as an absolute path on the box that ran it;
    only its basename is trusted, resolved against the run folder."""
    return directory / Path(summary['phases'][int(phase)]).name


def aligator_fence_snapshot(directory, summary, phase, call, sample):
    """The native (mesh, xs, us) of one captured sample of one native solve call, read
    exactly as the run's own `postprocess.py` reads it: sample -1 is the call's input
    (`input_<call>.npz`), sample k >= 0 is row k of the raw capture `trace_<call>.bin`
    (9 meta columns, then xs flattened, then the controls)."""
    phase_dir = _phase_dir(directory, summary, phase)
    initial = np.load(phase_dir / f'input_{int(call):02d}.npz')
    mesh = json.loads(str(initial['mesh_json']))
    offsets = initial['offsets']
    if int(sample) == -1:
        xs, controls = initial['xs'].copy(), initial['controls'].copy()
        source = phase_dir / f'input_{int(call):02d}.npz'
    else:
        record = json.loads((phase_dir / 'result.json').read_text())
        info = record['calls'][int(call)]
        if int(info['index']) != int(call):
            raise ValueError(f'{phase_dir}: call {call} is not at position {call} of result.json')
        source = phase_dir / f'trace_{int(call):02d}.bin'
        raw = np.memmap(source, dtype='<f8', mode='r', shape=(info['rows'], info['width']))
        if not 0 <= int(sample) < info['rows']:
            raise ValueError(f'{source}: sample {sample} outside 0..{info["rows"] - 1}')
        row = np.array(raw[int(sample)])
        ns = initial['xs'].size
        xs = row[9:9 + ns].reshape(initial['xs'].shape)
        controls = row[9 + ns:]
        del raw
    return mesh, xs, np.split(controls, offsets[1:-1]), source, phase_dir


def load_aligator_fence(robot, run_dir, snapshot=None, work_dir=None):
    """`snapshot=None`: the run's best-observed candidate (`best_observed_solution.npz`,
    the study's selection rule). A dict `{phase, call, sample}` (the `iterations.csv`
    stamps): that captured sample, whatever its violation (budget rollouts); extra keys
    of the dict are echoed into `settings['snapshot']`."""
    directory = Path(run_dir)
    if not (directory / 'best_observed_solution.npz').exists():
        directory = directory / robot
    summary = json.loads((directory / 'summary.json').read_text())
    best = summary['best_observed']
    if snapshot is None:
        config_dir = directory / '00_current_impact_cone'
    else:
        config_dir = _phase_dir(directory, summary, snapshot['phase'])
    config = json.loads((config_dir / 'config.json').read_text())
    if 'inputs' not in config:
        # a budget-extension phase continues its parent candidate on the same problem;
        # only the first phase records the problem / model inputs
        config = json.loads((_phase_dir(directory, summary, 0) / 'config.json').read_text())
    # The run's stored copies of its inputs (reference_inputs/).
    inputs = config['inputs']
    stored = _stored_inputs(inputs, directory / 'reference_inputs', 'numbered')
    problem = next(p for p in inputs if p.endswith(f'{robot}_jump_fence.yaml'))
    yaml_path, source, robot_files = _problem_yaml('fence', robot, stored, problem, work_dir)
    spec = load_fence_spec(yaml_path, allow_impact_cone=True)
    plan = fence_plan(spec, dt_target=.005, impulse_free=False)
    common = dict(problem_yaml=str(yaml_path), problem_yaml_source=str(source), problem_yaml_recorded=str(problem),
                  robot_files=robot_files, dt=.005)
    if snapshot is None:
        path = directory / 'best_observed_solution.npz'
        with np.load(path) as z:
            mesh = json.loads(str(z['mesh_json']))
            xs = z['xs'].copy()
            us = np.split(z['controls'], z['offsets'][1:-1])
        settings = dict(source=str(path),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), selected_record=best,
                        selection='best_observed_solution: minimum overall physical violation within solve budget',
                        gate=best['max_violation'], **common)
    else:
        mesh, xs, us, path, phase_dir = aligator_fence_snapshot(directory, summary, snapshot['phase'],
                                                                snapshot['call'], snapshot['sample'])
        settings = dict(source=str(path), snapshot=dict(snapshot), phase_dir=str(phase_dir),
                        selection='requested captured sample (phase, call, sample) of the recorded run',
                        gate=float(snapshot.get('node_max', float('nan'))), **common)
    r = spec.robot
    assert r.nq == r.nv and xs.shape == (len(mesh)+1, 2*r.nv)
    assert len(us) == len(mesh)
    if not all(np.isfinite(u).all() for u in us) or not np.isfinite(xs).all():
        raise ValueError(f'{path}: non-finite primal')
    ref = _new_reference(RecordedFenceReference, 'aligator', robot, path, directory/'summary.json', spec, plan, settings)
    _attach_native(ref, mesh, xs, us)
    assert np.isclose(mesh[-1]['t'] + mesh[-1]['h'], ref.ts[-1], atol=1e-10)
    _fill_grid(ref, xs)
    return ref


# ---------------------------------------------------------- Aligator terrain
def run_names(robot: str):
    return RUN_NAMES.get(robot, (robot,))


def find_run_dir(robot: str, archive_dir) -> Path | None:
    """`runs/<run name>/` of the experiment from the experiment folder, `runs/`, or the
    run folder itself; None when there is no best-observed npz there."""
    d = Path(archive_dir)
    for name in run_names(robot):
        for cand in (d / "runs" / name, d / name):
            if (cand / NPZ).exists() and (cand / "summary.json").exists():
                return cand
    if (d / NPZ).exists() and (d / "summary.json").exists():
        return d
    return None


def aligator_terrain_snapshot(directory: Path, phase: int, call: int, sample: int):
    """The native (mesh, xs, us) of one captured sample, read exactly as the run's own
    `terrain_benchmark_along_the_way/postprocess.py: snapshot_primal` reads it: sample -1
    is the call's input (`initial.npz` for call 0, else the last earlier finite call's
    returned `call_<k>.npz`), sample k >= 0 is row k of the raw capture
    `trace_<call>.primal.bin` (xs flattened, then the controls scaled by
    `settings.control_scale`)."""
    phase_dir = directory / f"phase_{int(phase):02d}"
    record = json.loads((phase_dir / "result.json").read_text())
    # The input of call k is the phase input for k = 0, else the returned primal of the
    # last earlier call that was finite (the run's `snapshot_primal`); the mesh and
    # control offsets are the phase's.
    initial_path = phase_dir / "initial.npz"
    for earlier in record["calls"][:int(call)]:
        if earlier.get("finite"):
            initial_path = phase_dir / f"call_{int(earlier['index']):02d}.npz"
    initial = np.load(initial_path)
    mesh = json.loads(str(initial["mesh_json"]))
    offsets = initial["offsets"]
    if int(sample) == -1:
        xs, controls, source = initial["xs"].copy(), initial["controls"].copy(), initial_path
    else:
        info = record["calls"][int(call)]["trace"]
        source = phase_dir / f"trace_{int(call):02d}.primal.bin"
        raw = np.memmap(source, dtype=np.float64, mode="r", shape=(info["count"], info["width"]))
        if not 0 <= int(sample) < info["count"]:
            raise ValueError(f"{source}: sample {sample} outside 0..{info['count'] - 1}")
        row = np.array(raw[int(sample)])
        nx = initial["xs"].size
        xs = row[:nx].reshape(initial["xs"].shape)
        controls = row[nx:] * float(record["settings"].get("control_scale", 1.0))
        del raw
    return mesh, xs, np.split(controls, offsets[1:-1]), source, phase_dir


def load_aligator_terrain(robot: str, run_dir, snapshot=None, work_dir=None):
    """`snapshot=None`: the run's best-observed candidate (`best_observed_solution.npz`,
    the study's selection rule). A dict `{phase, call, sample}` (the `iterations.csv`
    stamps): that captured sample, whatever its violation (budget rollouts); extra keys
    of the dict are echoed into `settings['snapshot']`."""
    directory = find_run_dir(robot, run_dir)
    if directory is None:
        raise FileNotFoundError(f"no {NPZ} for {robot} under {run_dir}")
    summary = json.loads((directory / "summary.json").read_text())
    if summary["robot"] not in run_names(robot):
        raise ValueError(f"Run robot {summary['robot']} is not a run of {robot} ({run_names(robot)})")
    best = summary["best_observed"]
    if snapshot is None:
        path = directory / NPZ
        with np.load(path) as z:
            mesh = json.loads(str(z["mesh_json"]))
            xs = z["xs"].copy()
            us = np.split(z["controls"], z["offsets"][1:-1])
            stamp = dict(phase=str(z["phase"]), call=int(z["call"]), sample=int(z["sample"]))
        phase_dir = directory / f"phase_{int(best['phase']):02d}"
        if Path(stamp["phase"]).name != phase_dir.name or stamp["call"] != int(best["call"]) \
                or stamp["sample"] != int(best["sample"]):
            raise ValueError(f"{path}: npz stamp {stamp} != summary best_observed "
                             f"(phase {best['phase']}, call {best['call']}, sample {best['sample']})")
    else:
        mesh, xs, us, path, phase_dir = aligator_terrain_snapshot(directory, snapshot["phase"], snapshot["call"],
                                                                  snapshot["sample"])
    config = json.loads((phase_dir / "config.json").read_text())
    # the run's stored copies (reference_inputs/<basename>)
    stored = _stored_inputs(config["inputs"], directory / "reference_inputs", "flat")
    name = next(n for n in config["inputs"] if n.endswith(".yaml") and "/problems/" in n)
    yaml_path, source, robot_files = _problem_yaml("terrain", robot, stored, name, work_dir)
    spec = load_terrain_spec(yaml_path)
    plan = terrain_plan(spec, dt_target=FINE_DT, kp=100.0, impulse_free=False)
    r = spec.robot
    nf = sum(c.dim for c in r.contacts)
    # xs[k] is the state entering mesh stage k; xs[-1] the independent terminal state.
    if r.nq != r.nv or xs.shape != (len(mesh) + 1, 2 * r.nv):
        raise ValueError(f"{path}: xs {xs.shape} != ({len(mesh) + 1}, {2 * r.nv})")
    if len(us) != len(mesh) or not all(np.isfinite(u).all() for u in us) or not np.isfinite(xs).all():
        raise ValueError(f"{path}: non-finite or mis-sized controls")
    flow_lengths = {len(us[i]) for i, s in enumerate(mesh) if s["kind"] == "flow"}
    if len(flow_lengths) != 1:
        raise ValueError(f"{path}: mixed control lengths {flow_lengths}")
    count = (flow_lengths.pop() - 3 * r.nv) // (r.nu + nf)
    if count not in (2, 3):
        raise ValueError(f"{path}: unknown control hold (count {count})")
    ts = np.asarray(plan.ts)
    if not np.isclose(mesh[-1]["t"] + mesh[-1]["h"], ts[-1], atol=1e-9):
        raise ValueError(f"{path}: duration {mesh[-1]['t'] + mesh[-1]['h']} != problem {ts[-1]}")
    if snapshot is None:
        settings = dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        selected_record=best, phase_dir=str(phase_dir),
                        selection="best_observed_solution: minimum recorded max(node_max, fine_max) over the "
                                  "budgeted run (summary.json['best_observed'])",
                        gate=float(best["max_violation"]), node_max=float(best["node_max"]),
                        fine_max=float(best["fine_max"]), passed_1e3=bool(best["max_violation"] <= 1e-3),
                        solve_s=float(best.get("time_s", np.nan)), termination=summary.get("termination"))
    else:
        settings = dict(source=str(path), snapshot=dict(snapshot), phase_dir=str(phase_dir),
                        selection="requested captured sample (phase, call, sample) of the recorded run",
                        gate=float(snapshot.get("node_max", np.nan)), termination=summary.get("termination"))
    settings.update(problem_yaml=str(yaml_path), problem_yaml_source=str(source), problem_yaml_recorded=name,
                    robot_files=robot_files, inputs=config["inputs"], dt=FINE_DT,
                    run_robot=summary["robot"], run_dir=str(directory),
                    hold="quadratic" if count == 3 else "linear", protocol=summary.get("protocol"),
                    aligator_version=config.get("aligator_version"), pinocchio_version=config.get("pinocchio_version"))
    ref = _new_reference(RecordedTerrainReference, "aligator", robot, path, directory / "summary.json", spec, plan, settings)
    _attach_native(ref, mesh, xs, us)
    _fill_grid(ref, xs)
    return ref


def reference_finite(ref, n=400) -> bool:
    """A quick finiteness screen of the reference the controller will sample."""
    for t in np.linspace(0.0, float(ref.duration), n):
        s = ref.eval(float(t))
        for arr in (s.q, s.v, s.a, s.u, *s.lam.values()):
            if not np.all(np.isfinite(arr)):
                return False
    return True


# ======================================================= budget-plan candidates
def candidate_dir_name(method, cid):
    """A filesystem name for a sample id: `seed`, `step_000123` (LEAP), `rec_000045`
    (Crocoddyl), `p00_c00_s000189` / `p00_c00_sm1` (Aligator `phase:call:sample`)."""
    if cid == "seed":
        return "seed"
    if method == "leap":
        return f"step_{int(float(cid)):06d}"
    if method == "crocoddyl":
        return f"rec_{int(cid):06d}"
    if method == "aligator":
        phase, call, sample = (int(x) for x in cid.split(":"))
        s = f"s{sample:06d}" if sample >= 0 else f"sm{-sample}"
        return f"p{phase:02d}_c{call:02d}_{s}"
    raise KeyError(method)


def load_study_plan(plan_path) -> dict:
    """A budget-candidate plan (`plan.json`, as `new_src/baselines/<method>/tools/derive_candidates.py --out`
    or `scripts/budget_study.py plan` writes it)."""
    return json.loads(Path(plan_path).read_text())


def baseline_series(plan_path=None, methods=METHODS, plan=None) -> list:
    """plan.json's series of the baselines, each with `run_dir` (the run folder = the
    parent of the series' `source`)."""
    plan = plan if plan is not None else load_study_plan(plan_path)
    return [dict(s, run_dir=str(Path(s["source"]).parent)) for s in plan["series"] if s["method"] in methods]


def budget_candidates(plan_path=None, methods=METHODS, plan=None) -> list:
    """Every baseline candidate of a plan (one rollout each): task, method, robot,
    candidate id, dir name, role, recorded time and node violation, the budgets that
    retain it, the run folder."""
    out = []
    for s in baseline_series(plan_path, methods, plan):
        for c in s["candidates"]:
            out.append(dict(task=s["task"], method=s["method"], robot=s["robot"], candidate=str(c["id"]), dir=c["dir"],
                            role=c.get("role", "incumbent"), recorded_time_s=c["recorded_time_s"],
                            node_violation=c["node_violation"], budgets=c["budgets"], run_dir=s["run_dir"]))
    return out


# ============================================================ one candidate
def load_baseline_reference(task: str, method: str, robot: str, candidate, *, run_dir, work_dir=None,
                            node_violation=None, recorded_time=None):
    """(reference, provenance) of one candidate of a baseline run.

    `run_dir`: the run folder `new_src/baselines/<method>/run.sh` wrote. `candidate`: a
    sample id (Crocoddyl '60', the record index -- on the terrain the `progress.csv` index,
    mapped to its stored candidate; Aligator 'phase:call:sample') or 'best' (the run's own
    selection rule); `new_src/baselines/<method>/README.md` shows how to print the run's
    final iterate id. `node_violation` / `recorded_time`: provenance only (the candidate's
    plan.json entry, when there is one)."""
    if method not in METHODS:
        raise KeyError(f"{method!r} is not a recorded baseline ({METHODS})")
    if task not in ("fence", "terrain"):
        raise KeyError(task)
    if run_dir is None:
        raise ValueError(f"run_dir is required: the {method} run folder written by new_src/baselines/{method}/run.sh")
    cid = str(candidate)
    if cid == "final":
        raise ValueError(f"'final' is not a candidate id: pass the run's final iterate id (new_src/baselines/{method}/README.md "
                         f"shows how to print it) or 'best'")
    run = Path(run_dir)
    if method == "crocoddyl":
        if cid == "best":
            loader = load_crocoddyl_fence if task == "fence" else load_crocoddyl_terrain
            ref = loader(robot, run, index=None, work_dir=work_dir)
            prov = dict(kind="selection_rule", selection=ref.settings["selection"],
                        record_index=ref.settings.get("record_index", ref.settings["snapshot_index"]),
                        stored_index=ref.settings.get("stored_index"))
        elif task == "fence":
            ref = load_crocoddyl_fence(robot, run, index=int(cid), work_dir=work_dir)
            prov = dict(kind="committed_record", record_index=int(cid))
        else:
            with (_run_dir_of(run, robot, "protocol.json") / "progress.csv").open(newline="") as f:
                rows = {int(r["index"]): r for r in csv.DictReader(f)}
            row = rows[int(cid)]
            stored = int(row["stored_index"])
            ref = load_crocoddyl_terrain(robot, run, index=stored, work_dir=work_dir)
            prov = dict(kind="stored_candidate", record_index=int(cid), stored_index=stored,
                        trajectory_changed=row["trajectory_changed"])
        prov.update(source=str(ref.npz_path), selected=ref.settings.get("selected_record"))
    else:
        snap = None
        if cid != "best":
            phase, call, sample = (int(x) for x in cid.split(":"))
            snap = dict(phase=phase, call=call, sample=sample)
            if node_violation is not None:
                snap["node_max"] = float(node_violation)
            if recorded_time is not None:
                snap["time_s"] = float(recorded_time)
        loader = load_aligator_fence if task == "fence" else load_aligator_terrain
        ref = loader(robot, run, snapshot=snap, work_dir=work_dir)
        if snap is None:
            b = ref.settings["selected_record"]      # summary.json's best_observed
            prov = dict(kind="selection_rule", selection=ref.settings["selection"],
                        phase=int(b["phase"]), call=int(b["call"]), sample=int(b["sample"]))
        else:
            prov = dict(kind="captured_sample", **snap)
        prov.update(source=str(ref.npz_path))
    prov.update(problem_yaml_source=ref.settings.get("problem_yaml_source"), task=task, method=method, robot=robot,
                candidate=cid, run_dir=str(run))
    ref.settings["candidate"] = cid
    return ref, prov


def track_recorded(task: str, robot: str, method: str, out_dir, candidate, *, run_dir,
                   hold=1.5, settle=0.05, clock="touchdown", early_jump=None, gains=None, verbose=False,
                   control_dt=None, duration=None, friction=None, capsule_collision=True, approach_ring=0.05,
                   evaluate_after=True, node_violation=None, recorded_time=None):
    """`rollout.track_archive` for a recorded baseline candidate: the same protocol, the same
    `rollout.run` (gains and options of the task and robot, `controllers.gains_for` /
    `options_for`), the same evaluator, the same files in `out_dir` (rollout.npz,
    metrics.json, trajectory.npz, evaluation.json, evaluation.npz, run.log, the problem
    yaml), with `method` = crocoddyl / aligator. Returns the metrics dict (with
    `evaluation`). Raises when the reference cannot be built or is not finite."""
    import contextlib
    import sys
    import time
    import mujoco
    from .controllers import gains_for, options_for
    from .evaluate import evaluate_dir
    from .rollout import _Tee, _json, from_rollout, print_summary, run

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = options_for(task, dict(hold=hold, settle=settle, control_dt=control_dt, early_jump=early_jump), robot=robot)
    gains = gains or gains_for(task, robot)
    header = dict(task=task, robot=robot, method=method, candidate=str(candidate),
                  run_dir=(str(run_dir) if run_dir else None), clock=clock, friction=friction, duration=duration,
                  capsule_collision=capsule_collision, approach_ring=approach_ring, options=opts,
                  gains=gains.to_dict(), python=sys.executable, mujoco=mujoco.__version__)
    with (out_dir / "run.log").open("w") as lg:
        lg.write("leaptrack.track_recorded " + json.dumps(header, default=_json) + "\n")
        lg.flush()
        streams = (lg, sys.stdout) if verbose else (lg,)
        with contextlib.redirect_stdout(_Tee(*streams)):
            t0 = time.perf_counter()
            ref, prov = load_baseline_reference(task, method, robot, candidate, run_dir=run_dir, work_dir=out_dir,
                                                node_violation=node_violation, recorded_time=recorded_time)
            print(f"reference: {method} {task}/{robot} candidate {prov['candidate']} ({prov['kind']}) from "
                  f"{prov['source']}, recorded violation {ref.settings.get('gate')}, T = {ref.duration:.4f} s")
            if not reference_finite(ref):
                raise ValueError(f"{prov['source']}: non-finite reference samples")
            out, metrics, ref, rs = run(task, robot, reference=ref, out_dir=None, gains=gains, clock=clock,
                                        duration=duration, friction=friction, capsule_collision=capsule_collision,
                                        approach_ring=approach_ring, verbose=False, **opts)
            # `run` stamps the LEAP loader's method on the metrics; this rollout tracked a baseline
            metrics.update(method=ref.method, candidate=prov["candidate"], baseline=prov)
            print_summary(metrics)
            np.savez_compressed(out_dir / "rollout.npz", **out)
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=_json))
            traj = from_rollout(out, metrics, ref, benchmark=task, method=ref.method,
                                extra_meta=dict(problem_yaml=metrics["problem_yaml"], candidate=prov["candidate"],
                                                baseline=prov))
            traj.save(out_dir / "trajectory.npz")
            print(f"wrote {out_dir / 'rollout.npz'}  {out_dir / 'metrics.json'}  {out_dir / 'trajectory.npz'}")
            if evaluate_after and duration is None:
                metrics["evaluation"] = evaluate_dir(out_dir, task=task, verbose=True, ref=ref)
            print(f"total wall {time.perf_counter() - t0:.0f} s")
    return metrics


def reference_of_case(case_dir, work_dir=None):
    """Reload the reference a baseline rollout tracked, from its trajectory metadata (to
    re-evaluate it: `evaluate.evaluate_dir(case_dir, ref=reference_of_case(case_dir))`;
    `evaluate_dir` alone reloads LEAP archives only)."""
    from .rollout import TrackedTrajectory
    b = TrackedTrajectory.load(Path(case_dir) / "trajectory.npz").meta["baseline"]
    ref, _ = load_baseline_reference(b["task"], b["method"], b["robot"], b["candidate"], run_dir=b["run_dir"],
                                     work_dir=work_dir or case_dir, node_violation=b.get("node_max"),
                                     recorded_time=b.get("time_s"))
    return ref


# ======================================================================== CLI
def main(argv=None):
    """python -m leaptrack.recorded show --task T --method M --robot R --run-dir DIR --candidate C"""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("what", choices=("show",))
    ap.add_argument("--task", choices=("fence", "terrain"), required=True)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--run-dir", required=True, help="the run folder written by new_src/baselines/<method>/run.sh")
    ap.add_argument("--candidate", required=True, help="a sample id of the run, or best")
    a = ap.parse_args(argv)
    ref, prov = load_baseline_reference(a.task, a.method, a.robot, a.candidate, run_dir=a.run_dir)
    print(json.dumps({k: v for k, v in prov.items() if k != "selected"}, indent=1, default=str))
    print(f"T={ref.duration:.4f}s nq={ref.nq} nu={ref.nu} contacts={ref.contact_dims} planarlock={ref.planarlock} "
          f"mu={ref.spec.mu} impacts={ref.impact_times}")
    print("phase times", np.round(ref.phase_times, 4))
    print("touchdown events", [(round(t, 4), f) for t, f in ref.touchdown_events])
    for t in [0.0] + [x - 1e-4 for x in ref.impact_times] + [x + 1e-4 for x in ref.impact_times] + [ref.duration + 0.3]:
        s = ref.eval(t)
        print(f"  t={t:.4f} phase={ref.phase_names[s.phase]}({s.phase_s:.2f}) sched={s.scheduled} base={np.round(s.q[:3], 3)} "
              f"vz={s.v[2]:+.3f} |u|={np.linalg.norm(s.u):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

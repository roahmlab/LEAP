"""Table III's quantity `max_i e_i / e_bar_i` of a tracked rollout. Standard library only.

The ratio needs only `evaluation.json["final_state"]["end" | "at_T"]` with the
six errors:

    base_pos_err 0.050 m, base_rot_err 0.10 rad, joint_err_max 0.20 rad, foot_err_max 0.040 m,
    base_speed 0.10 m/s, joint_speed_max 0.10 rad/s

evaluated at the end of the 1.5 s hold ("end"), except fence hopper2d at the end
of the plan ("at_T"). A value below 1 means the rollout reaches the goal.
"""
from __future__ import annotations

import json
from pathlib import Path

THRESHOLDS = {"base_pos_err": 0.050, "base_rot_err": 0.10, "joint_err_max": 0.20,
              "foot_err_max": 0.040, "base_speed": 0.10, "joint_speed_max": 0.10}
JUDGE_AT_T = {("fence", "hopper2d")}


def judge_tag(task: str, robot: str) -> str:
    return "at_T" if (task, robot) in JUDGE_AT_T else "end"


def terminal_ratio(evaluation, task: str, robot: str):
    """`evaluation`: an evaluation.json path or its dict. Returns
    (tag, ratio, worst_metric, {metric: e_i / e_bar_i})."""
    if not isinstance(evaluation, dict):
        evaluation = json.loads(Path(evaluation).read_text())
    tag = judge_tag(task, robot)
    block = evaluation["final_state"][tag]
    ratios = {k: float(block[k]) / t for k, t in THRESHOLDS.items()}
    worst = max(ratios, key=ratios.get)
    return tag, ratios[worst], worst, ratios


def ratio(evaluation, task: str, robot: str):
    """score_terminal.py's signature: (tag, ratio, worst)."""
    tag, r, worst, _ = terminal_ratio(evaluation, task, robot)
    return tag, r, worst

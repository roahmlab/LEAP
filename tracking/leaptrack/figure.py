"""Fig. 3: rollout terminal error against compute budget, both benchmarks in one printed
column (terrain | fence, a robot per row, one shared budget axis).

The LEAP curves come from the budget study (`budget.collect` -> results.json). A
Crocoddyl or Aligator series with at least one candidate rolled out in the same study
(`budget_study.py plan --baselines`) is drawn from the study's rows; otherwise it can be
drawn from another study's `results.json` (`plot --baselines`), whose rows carry the same
`terminal_ratio_end` / `_at_T` scalars (`source` in the summary .json).

At budget b the step value is the terminal-goal quantity max_i e_i / e_bar_i of the
retained candidate's rollout, at the end of the 1.5 s hold, or at the end of the plan for
the cases judged there (fence hopper2d). A dashed line marks 1, the success criterion.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_STYLE = HERE / "style.json"
JUDGE_AT_T = {("fence", "hopper2d")}
OK_STATUSES = ("ok", "unstable")
COLUMNS = (dict(key="terrain"), dict(key="fence"))

# The figure is drawn at SCALE times the width it is printed at (one IEEE column, 3.5 in),
# so every size below is stated in final printed points and multiplied once.
SCALE = 2.0
COLUMN_WIDTH_IN = 3.5
PT = dict(tick=6.0, panel=6.5, axis=7.0, title=8.0)


def load_style(path=DEFAULT_STYLE):
    return json.loads(Path(path).read_text())


def apply_matplotlib(config):
    import matplotlib as mpl
    from cycler import cycler
    mpl.rcParams.update(config["matplotlib"])
    mpl.rcParams["axes.prop_cycle"] = cycler(color=[v["rgba"][:3] for v in config["methods"].values()])


def judged_at_T(task, robot):
    return (task, robot) in JUDGE_AT_T


def series_curve(plan_series, rows, key_end="terminal_ratio_end", key_at_T="terminal_ratio_at_T"):
    """The step curve of one (task, method, robot): at every planned budget the metric of
    the retained candidate's rollout, NaN where that rollout has no evaluation."""
    task, method, robot = plan_series["task"], plan_series["method"], plan_series["robot"]
    key = key_at_T if judged_at_T(task, robot) else key_end
    t, y, failed, cands = [], [], [], []
    for b in plan_series["budgets"]:
        cid = b["candidate"]
        row = rows.get((task, method, robot, cid))
        val, bad = np.nan, True
        if row is not None and row.get("status") in OK_STATUSES:
            v = row.get(key)
            if v is not None and np.isfinite(v):
                val, bad = float(v), False
        t.append(float(b["budget_s"]))
        y.append(val)
        failed.append(bad)
        cands.append(cid)
    t, y = np.asarray(t, float), np.asarray(y, float)
    final = None
    fid = plan_series.get("final_iterate")
    if fid is not None and fid != cands[-1]:
        row = rows.get((task, method, robot, fid))
        if row is not None and row.get("status") in OK_STATUSES and row.get(key) is not None:
            final = dict(candidate=fid, value=float(row[key]))
    return dict(time=t, best=y, failed=np.asarray(failed, bool), candidates=cands,
                last_time_s=float(plan_series["last_time_s"]), final_iterate=final)


def load_results(path):
    data = json.loads(Path(path).read_text())
    plan = json.loads(Path(data["plan"]).read_text())
    rows = {(r["task"], r["method"], r["robot"], r["candidate"]): r for r in data["rows"]}
    return plan, rows


def build_columns(config, plan, rows, methods, baselines=None):
    """columns: per task the robots (ascending joint count), the curves per (robot, method),
    the labels and joint counts. `baselines` = (plan, rows) of another study with more
    methods, read the same way."""
    by_key = {(s["task"], s["method"], s["robot"]): s for s in plan["series"]}
    # a baseline series of this study counts once any of its candidates was rolled out here;
    # a series with nothing evaluated here is taken from `baselines` if given
    rolled = {k[:3] for k, r in rows.items() if r.get("status") != "pending"}
    by_key = {k: s for k, s in by_key.items() if k[1] == "leap" or k in rolled}
    sources = {k: "this study" for k in by_key if k[1] != "leap"}
    if baselines is not None:  # the other methods' curves only: the LEAP curve is always this study's
        bplan, brows = baselines
        for s in bplan["series"]:
            key = (s["task"], s["method"], s["robot"])
            if s["method"] != "leap" and key not in by_key:
                by_key[key] = s
                sources[key] = "baselines"
        # a series drawn from `baselines` takes only its rows there (this study's pending rows of the
        # same recorded ids must not shadow them)
        rows = {**{k: v for k, v in brows.items() if k[1] != "leap" and sources.get(k[:3]) == "baselines"},
                **{k: v for k, v in rows.items() if sources.get(k[:3]) != "baselines"}}
    columns = []
    for spec in COLUMNS:
        section = config["tasks"][spec["key"]]
        counts = section["robot_joint_counts"]
        robots = sorted(section["robots"], key=counts.__getitem__)
        data = {}
        for r in robots:
            for m in methods:
                s = by_key.get((spec["key"], m, r))
                if s is not None:
                    data[r, m] = series_curve(s, rows)
        columns.append(dict(spec, title=section["title"], robots=robots, data=data, counts=dict(counts), labels=section["robot_labels"],
                            sources={r_m: sources[(spec["key"], r_m[1], r_m[0])] for r_m in data
                                     if (spec["key"], r_m[1], r_m[0]) in sources}))
    return columns


def fill_failures(columns, factor=3.0):
    """A candidate whose rollout could not be evaluated has no value to plot; it is drawn
    at `factor` times the largest evaluated value of the figure and marked with a cross."""
    finite = np.concatenate([d["best"][np.isfinite(d["best"])] for c in columns for d in c["data"].values()])
    top = float(finite.max()) * factor if finite.size else 1.0
    for c in columns:
        for d in c["data"].values():
            d["best"] = np.where(np.isfinite(d["best"]), d["best"], top)
    return top


def draw(config, columns, methods, xscale="log", ylabel=r"Rollout terminal error $\max_i\, e_i/\bar{e}_i$",
         xlabel="Recorded compute budget [s]", y_major_base=100.0, label_corner="bottom left"):
    """Ten panels in ONE printed column: a robot per row, a task per column, one shared
    budget axis, one violation axis per row, no legend (the caption names the colours)."""
    rows = len(columns[0]["robots"])
    sizes = {k: v * SCALE for k, v in PT.items()}
    fig, cells = plt.subplots(rows, len(columns), figsize=(COLUMN_WIDTH_IN * SCALE, 4.3 * SCALE), sharex=True, sharey="row")
    every = [d for c in columns for d in c["data"].values()]
    positive = np.concatenate([d["time"][d["time"] > 0] for d in every])
    left = float(positive.min()) if xscale == "log" else 0.0
    right = max(float(d["time"][-1]) for d in every)
    for j, column in enumerate(columns):
        for i, robot in enumerate(column["robots"]):
            ax = cells[i, j]
            for method in methods:
                d = column["data"].get((robot, method))
                if d is None:
                    continue
                color = config["methods"][method]["rgba"]
                t = np.maximum(d["time"].copy(), left)  # a zero-time seed sits at the first positive tick
                shown = np.maximum(d["best"], 1e-12)
                ax.step(t, shown, where="post", color=color, label=config["methods"][method]["label"], linewidth=0.9 * SCALE)
                # the dot: where the run ended; the dotted tail: its final iterate kept beyond
                k = int(np.searchsorted(d["time"], d["last_time_s"], side="right") - 1)
                k = max(0, min(k, len(t) - 1))
                ax.plot(t[k], shown[k], "o", color=color, ms=1.6 * SCALE)
                if t[k] < right:
                    ax.plot([t[k], right], [shown[-1]] * 2, color=color, linestyle=":", linewidth=0.7 * SCALE)
            ax.set_xscale(xscale)
            ax.set(yscale="log", xlim=(left, right * 1.04))
            if xscale == "log":
                ax.xaxis.set_major_locator(LogLocator(base=100.0))
                ax.xaxis.set_minor_locator(LogLocator(base=10.0))
                ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(LogLocator(base=y_major_base))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(labelsize=sizes["tick"], length=2.0 * SCALE, width=0.5 * SCALE, pad=1.5 * SCALE)
            ax.tick_params(which="minor", length=1.0 * SCALE, width=0.4 * SCALE)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5 * SCALE)
            ax.grid(True, which="major", alpha=0.2, linewidth=0.5 * SCALE)
            text = f"{column['labels'][robot]} · {column['counts'][robot]} joints"
            if label_corner == "bottom left":
                ax.text(0.03, 0.08, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=sizes["panel"])
            else:
                ax.text(0.97, 0.92, text, transform=ax.transAxes, ha="right", va="top", fontsize=sizes["panel"])
            if i == 0:
                ax.set_title(column["title"], loc="center", fontsize=sizes["title"], pad=3.0 * SCALE)
            ax.axhline(1.0, color="0.35", linestyle="--", linewidth=0.6 * SCALE, zorder=0)  # the success criterion
    fig.supylabel(ylabel, fontsize=sizes["axis"], x=0.015)
    fig.tight_layout(rect=(0.015, 0, 1, 1), h_pad=0.5, w_pad=1.2)
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    floor = min(ax.get_tightbbox(fig.canvas.get_renderer()).transformed(to_fig).y0 for ax in cells[-1, :])
    fig.text(0.57, floor - 0.012, xlabel, ha="center", va="top", fontsize=sizes["axis"])
    return fig


def mark_failures(fig, config, columns, methods, top):
    axes = np.asarray(fig.axes).reshape(len(columns[0]["robots"]), len(columns))
    for j, column in enumerate(columns):
        for i, robot in enumerate(column["robots"]):
            for method in methods:
                d = column["data"].get((robot, method))
                if d is None:
                    continue
                bad = d["failed"] & (d["best"] >= top * (1 - 1e-9))
                if bad.any():
                    left = axes[i, j].get_xlim()[0]
                    axes[i, j].plot(np.maximum(d["time"][bad], left), d["best"][bad], "x", color=config["methods"][method]["rgba"],
                                    ms=2.2 * SCALE, mew=0.6 * SCALE)


def legend_fragment(config, methods):
    lines = ["% Generated by new_src/tracking/leaptrack/figure.py.", "% Do not edit: colours and names come from leaptrack/style.json."]
    names = []
    for method in methods:
        entry = config["methods"][method]
        red, green, blue = entry["rgba"][:3]
        lines.append(f"\\definecolor{{method{method}}}{{rgb}}{{{red:.3f},{green:.3f},{blue:.3f}}}")
        names.append(f"\\textcolor{{method{method}}}{{{entry['label']}}} ({entry['color_name']})")
    key = (", ".join(names[:-1]) + " and " + names[-1]) if len(names) > 1 else names[0]
    lines.append("\\newcommand{\\budgetviolationkey}{%s}" % key)
    return "\n".join(lines) + "\n"


def make_figure(results_json: Path, out_dir: Path, style=DEFAULT_STYLE, baselines_json: Path | None = None, stem=None,
                hopper_joints: int = 2):
    config = load_style(style)
    apply_matplotlib(config)
    plan, rows = load_results(results_json)
    methods = ["leap"]
    baselines = None
    if baselines_json is not None:
        bplan, brows = load_results(baselines_json)
        baselines = (bplan, brows)
    own = {k[1] for k, r in rows.items() if r.get("status") != "pending"}   # methods rolled out by this study
    for m in config["methods"]:
        if m != "leap" and (m in own or (baselines is not None and any(k[1] == m for k in baselines[1]))):
            methods.append(m)
    columns = build_columns(config, plan, rows, methods, baselines)
    if hopper_joints is not None and hopper_joints >= 0:
        # both hoppers are labelled with `hopper_joints` joints
        for column in columns:
            for robot in column["robots"]:
                if robot.startswith("hopper2d"):
                    column["counts"][robot] = hopper_joints
    top = fill_failures(columns)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or "rollout_violation_under_time_budget_combined_normalized_terminal_log"
    fig = draw(config, columns, methods)
    mark_failures(fig, config, columns, methods, top)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{stem}.{suffix}")
    plt.close(fig)
    (out_dir / f"{stem}_key.tex").write_text(legend_fragment(config, methods))
    with (out_dir / f"{stem}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("task", "robot", "method", "budget_s", "candidate", "rollout_metric", "evaluated"))
        for column in columns:
            for robot in column["robots"]:
                for method in methods:
                    d = column["data"].get((robot, method))
                    if d is None:
                        continue
                    for k in range(len(d["time"])):
                        w.writerow((column["key"], robot, method, d["time"][k], d["candidates"][k],
                                    "" if d["failed"][k] else d["best"][k], not d["failed"][k]))
    # "earliest trackable budget": the first budget from which the retained reference stays
    # below 1 to the end of the run
    summary = []
    for column in columns:
        for robot in column["robots"]:
            for method in methods:
                d = column["data"].get((robot, method))
                if d is None:
                    continue
                ok = ~d["failed"]
                below = ok & (d["best"] < 1.0)
                earliest = None
                for k in range(len(d["time"])):
                    if below[k:].all():
                        earliest = float(d["time"][k])
                        break
                summary.append(dict(task=column["key"], robot=robot, method=method, evaluated=int(ok.sum()), budgets=int(len(d["time"])),
                                    earliest_trackable_budget_s=earliest, final_rollout_metric=float(d["best"][-1]) if ok[-1] else None,
                                    judged_at_T=judged_at_T(column["key"], robot)))
                if method != "leap":     # where the baseline curve came from
                    summary[-1]["source"] = column["sources"].get((robot, method))
    (out_dir / f"{stem}.json").write_text(json.dumps(dict(results=str(results_json), baselines=str(baselines_json) if baselines_json else None,
                                                             methods=methods, series=summary), indent=2) + "\n")
    return out_dir / f"{stem}.pdf", summary

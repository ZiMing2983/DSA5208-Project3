#!/usr/bin/env python3
"""Report figures from the result CSVs (PROJECT_PLAN.md §6, §10).

    python3 analysis/plot_scaling.py [CSV ...] [--out report/figures]

Reads results/*.csv by default, takes the median of repeated runs and writes one PDF and one
PNG per figure. Axes for time, problem size and core count are logarithmic and every scaling
plot has its ideal line, following the instructor's plot rules. Figures without data are
skipped with a note.
  e1_<machine>        serial time/iteration vs n, iterations vs N, time to solution vs n
  strong_<machine>    E3: time/iteration, speedup and efficiency vs p
  weak_<machine>      E4: time/iteration and weak-scaling efficiency vs p
  comm_<machine>_p<P> E5: compute / halo / reduction time per iteration of each variant
  hybrid_<machine>    E6: time/iteration for each ranks x threads split
  placement_<machine> rank placement on one node (--map-by core/socket/numa/l3cache)
  gpu_<machine>       E7: 1 core vs 1 CPU node vs 1 and 2 GPUs, and achieved bandwidth
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import amdahl_fit, load, median_runs, serial_baseline, size_label  # noqa: E402

A40_PEAK_GBS = 696.0


def plain_ticks(ax, xticks=None):
    """Plain numbers (2, 64, 0.5) instead of 2x10^0 on log axes; optional explicit x ticks."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
        if scale == "log":
            axis.set_major_formatter(fmt)
            axis.set_minor_formatter(NullFormatter())
    if xticks is not None:
        # Label a tick only if it is at least 1.3x the previous labelled one (64 and 72 would
        # overlap on a log axis); the others become unlabelled minor ticks.
        ticks = sorted(set(int(v) for v in xticks))
        shown = []
        for t in ticks:
            if not shown or t >= 1.3 * shown[-1]:
                shown.append(t)
        ax.set_xticks(shown)
        ax.set_xticks([t for t in ticks if t not in shown], minor=True)
        ax.xaxis.set_major_formatter(fmt)


def log_yticks(ax):
    """Labelled ticks at 1, 2, 3, 5 x 10^k inside the y range of a log axis, so that a narrow
    range (30-40 ms) still has labels: a log axis labels only powers of ten by default."""
    lo, hi = ax.get_ylim()
    ticks = [m * 10.0 ** k for k in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1)
             for m in (1, 2, 3, 5)]
    ticks = [t for t in ticks if lo <= t <= hi]
    if len(ticks) < 3:  # a narrow range: evenly spaced round numbers instead
        ticks = [t for t in MaxNLocator(nbins=6).tick_values(lo, hi) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(lo, hi)


def mark_nodes(axes, df):
    """A dotted line at the largest single-node rank count, when there are multi-node points."""
    if df.nodes.max() <= 1 or not (df.nodes == 1).any():
        return
    edge = df[df.nodes == 1].p.max()
    for i, a in enumerate(axes):
        a.axvline(edge, color="grey", ls=":", lw=1)
        if i == 0:
            a.text(edge, 0.02, f" 1 node | {int(df.nodes.max())} nodes max", color="grey", fontsize=7,
                   transform=a.get_xaxis_transform(), ha="left", va="bottom")


def save(fig, out, name):
    fig.set_layout_engine("constrained")
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/{name}.pdf/.png")


def slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return np.polyfit(x, y, 1)[0] if len(x) >= 2 else float("nan")


def plot_e1(med, machine, out):
    fixed = med[(med.machine == machine) & (med.backend == "serial") & (med.tag == "e1")]
    tol = med[(med.machine == machine) & (med.backend == "serial") & (med.tag == "e1-tol")]
    if fixed.empty and tol.empty:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for dim, marker in ((2, "o"), (3, "s")):
        f = fixed[fixed.dim == dim].sort_values("n")
        if len(f):
            ax[0].loglog(f.n, f.t_iter, marker + "-",
                         label=f"{dim}D (slope {slope(f.n, f.t_iter):.2f})")
        t = tol[tol.dim == dim].sort_values("n")
        if len(t):
            ax[1].loglog(t.nx, t.iters, marker + "-",
                         label=f"{dim}D (slope {slope(t.nx, t.iters):.2f}, theory 1)")
            ax[2].loglog(t.n, t.t_total, marker + "-",
                         label=f"{dim}D (slope {slope(t.n, t.t_total):.2f}, theory {1 + 1 / dim:.2f})")
    if len(fixed):
        n = np.array([fixed.n.min(), fixed.n.max()], float)
        ref = fixed.sort_values("n").iloc[-1]
        ax[0].loglog(n, ref.t_iter * n / ref.n, "k--", lw=1, label="slope 1 (linear in n)")
    ax[0].set(xlabel="unknowns n", ylabel="time per CG iteration [s]",
              title=f"E1 serial time per iteration ({machine})")
    ax[1].set(xlabel="grid points per dimension N", ylabel="CG iterations to tol 1e-8",
              title="E1 iterations (O(N), kappa ~ N^2)")
    ax[2].set(xlabel="unknowns n", ylabel="time to solution [s]", title="E1 time to solution")
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
    plain_ticks(ax[1], sorted(tol.nx.unique()) if len(tol) else None)
    save(fig, out, f"e1_{machine}")


def plot_strong(med, machine, out):
    s = med[(med.machine == machine) & (med.tag == "strong") & (med.threads == 1)]
    s = s[(s.decomp == "block") & (s.halo == "nonblocking") & (s.cg == "standard") &
          (s["dot"] == "fast")]
    if s.empty:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for (src, dim, nx, ny, nz), grp in s.groupby(["source", "dim", "nx", "ny", "nz"]):
        grp = grp.sort_values("p")
        t1 = serial_baseline(med, machine, dim, nx, ny, nz, source=src)
        base = "serial"
        if t1 is None:
            p1 = grp[grp.p == 1]
            if p1.empty:
                continue
            t1, base = float(p1.t_iter.iloc[0]), "MPI p=1"
        label = size_label(grp.iloc[0])
        sp = t1 / grp.t_iter
        line, = ax[0].loglog(grp.p, grp.t_iter, "o-", label=label)
        ax[0].loglog(grp.p, t1 / grp.p, "--", color=line.get_color(), lw=0.8)
        ax[1].loglog(grp.p, sp, "o-", color=line.get_color(), label=f"{label} (T1: {base})")
        f = amdahl_fit(grp.p, sp)
        if f == f and f > 0:  # Amdahl's law with the fitted serial fraction (Lecture 7)
            pp = np.geomspace(1, grp.p.max(), 50)
            ax[1].loglog(pp, 1 / (f + (1 - f) / pp), ":", color=line.get_color(), lw=1,
                         label=f"Amdahl fit, f = {f:.3f}")
        ax[2].semilogx(grp.p, sp / grp.p, "o-", color=line.get_color(), label=label)
    pmax = s.p.max()
    ax[1].loglog([1, pmax], [1, pmax], "k--", lw=1, label="ideal")
    ax[2].axhline(1.0, color="k", ls="--", lw=1)
    ax[0].set(xlabel="MPI ranks p", ylabel="time per iteration [s]",
              title=f"E3 strong scaling ({machine}); dashed: ideal T1/p")
    ax[1].set(xlabel="MPI ranks p", ylabel="speedup T1 / Tp", title="speedup")
    ax[2].set(xlabel="MPI ranks p", ylabel="parallel efficiency Sp / p", title="efficiency")
    mark_nodes(ax, s)
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=7)
        plain_ticks(a, s.p.unique())
    save(fig, out, f"strong_{machine}")


def plot_weak(med, machine, out):
    w = med[(med.machine == machine) & (med.tag == "weak") & (med.threads == 1)]
    if w.empty:
        return
    w = w.assign(local=(w.nx // w.px).astype(int))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for (dim, local), grp in w.groupby(["dim", "local"]):
        grp = grp.sort_values("p")
        ref = grp[grp.p == 1]
        if ref.empty:
            ref = grp.iloc[[0]]
        t1 = float(ref.t_iter.iloc[0])
        label = f"{dim}D, {local}^{dim} per rank"
        line, = ax[0].loglog(grp.p, grp.t_iter * 1e3, "o-", label=label)
        ax[0].axhline(t1 * 1e3, color=line.get_color(), ls="--", lw=0.8)
        ax[1].semilogx(grp.p, t1 / grp.t_iter, "o-", color=line.get_color(), label=label)
        ax[2].loglog(grp.p, grp.p * t1 / grp.t_iter, "o-", color=line.get_color(), label=label)
    ax[1].axhline(1.0, color="k", ls="--", lw=1, label="ideal")
    ax[2].loglog([1, w.p.max()], [1, w.p.max()], "k--", lw=1, label="ideal (= p)")
    ax[2].set(xlabel="MPI ranks p", ylabel="scaled speedup p T(1)/T(p)",
              title="Gustafson scaled speedup")
    ax[0].set(xlabel="MPI ranks p", ylabel="time per iteration [ms]",
              title=f"E4 weak scaling ({machine}); dashed: ideal (flat)")
    ax[1].set(xlabel="MPI ranks p", ylabel="weak efficiency T(1)/T(p)", title="weak efficiency")
    mark_nodes(ax, w)
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
        plain_ticks(a, w.p.unique())
    log_yticks(ax[0])
    save(fig, out, f"weak_{machine}")


def plot_comm(med, machine, out):
    """A dot plot on a log time axis: the total and each phase of every variant (a stacked bar
    would need a linear axis, and would hide the sub-millisecond phases under the stencil's)."""
    c = med[(med.machine == machine) & (med.tag == "comm")]
    for (p, dim, nx), grp in c.groupby(["p", "dim", "nx"]):
        grp = grp.sort_values("t_iter", ascending=False)  # fastest variant at the top
        labels = [f"{r.decomp} / {r.halo}" + (" / fused CG" if r.cg == "fused" else "") +
                  (" / repro dots" if r.dot == "repro" else "") for r in grp.itertuples()]
        y = np.arange(len(grp))
        nodes = int(grp.nodes.max())
        fig, ax = plt.subplots(figsize=(8, 0.3 * len(grp) + 1.6), layout="constrained")
        for col, lab, mk, colour in (("t_iter", "total (slowest rank)", "o", "black"),
                                     ("comp_per_iter", "stencil + vector updates", "^", "#4C72B0"),
                                     ("halo_per_iter", "halo exchange", "s", "#DD8452"),
                                     ("reduce_per_iter", "Allreduce", "D", "#55A868")):
            ax.semilogx(grp[col].to_numpy() * 1e3, y, mk, color=colour, ms=5, ls="none", label=lab)
        ax.set_yticks(y, labels, fontsize=8)
        ax.set_ylim(-0.7, len(grp) - 0.3)
        ax.set(xlabel="time per iteration [ms] (phases averaged over ranks)",
               title=f"E5 communication variants, {size_label(grp.iloc[0])}, p = {p} on "
                     f"{nodes} node{'s' if nodes > 1 else ''} ({machine})")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.grid(True, axis="x", which="both", alpha=0.3)
        fig.legend(fontsize=8, ncol=4, loc="outside lower center")
        save(fig, out, f"comm_{machine}_{dim}d_p{p}")


def plot_hybrid(med, machine, out):
    h = med[(med.machine == machine) & (med.tag == "hybrid")]
    if h.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for (cores, nodes), grp in h.groupby(["cores", "nodes"]):
        grp = grp.sort_values("threads")
        ax.loglog(grp.threads, grp.t_iter * 1e3, "o-",
                  label=f"{cores} cores on {nodes} node{'s' if nodes > 1 else ''} "
                        f"({size_label(grp.iloc[0])})")
        for r in grp.itertuples():  # ranks per node x threads
            ax.annotate(f"{r.p // nodes}x{r.threads}", (r.threads, r.t_iter * 1e3), fontsize=7,
                        textcoords="offset points", xytext=(3, 4))
    ax.set(xlabel="OpenMP threads per rank (ranks per node x threads = cores per node)",
           ylabel="time per iteration [ms]", title=f"E6 hybrid MPI+OpenMP ({machine})")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo / 1.15, hi * 1.15)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plain_ticks(ax, h.threads.unique())
    log_yticks(ax)
    save(fig, out, f"hybrid_{machine}")


def plot_placement(med, machine, out):
    pl = med[(med.machine == machine) & med.tag.str.startswith("placement-")]
    if pl.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for tag, grp in pl.groupby("tag"):
        grp = grp.sort_values("p")
        ax.loglog(grp.p, grp.t_iter * 1e3, "o-", label=tag.split("-", 1)[1])
    full = med[(med.machine == machine) & (med.tag == "strong") & (med.dim == 3) &
               (med.nx == 512) & (med.threads == 1)]
    if len(full):
        best = full.sort_values("p").iloc[-1]
        ax.axhline(best.t_iter * 1e3, color="k", ls="--", lw=1,
                   label=f"all {int(best.p)} cores (memory-bandwidth limit)")
    ax.set(xlabel="MPI ranks p on one node", ylabel="time per iteration [ms]",
           title=f"Rank placement within one node, {size_label(pl.iloc[0])} ({machine})")
    plain_ticks(ax, pl.p.unique())
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="mpirun --map-by", fontsize=8)
    save(fig, out, f"placement_{machine}")


def plot_gpu(med, machine, out):
    g = med[(med.machine == machine) & (med.tag == "gpu-strong")]
    if g.empty:
        return
    problems = g.groupby(["dim", "nx", "ny", "nz"])
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    width = 0.11
    names = []
    labelled = set()
    for i, ((dim, nx, ny, nz), grp) in enumerate(problems):
        names.append(size_label(grp.iloc[0]))
        bars = []
        t1 = serial_baseline(med, machine, dim, nx, ny, nz)
        bars.append(("1 core (serial)", t1))
        cpu = med[(med.machine == machine) & (med.tag == "strong") & (med.dim == dim) &
                  (med.nx == nx) & (med.ny == ny) & (med.nz == nz) & (med.threads == 1)]
        for nodes in (1, 4):
            nd = cpu[cpu.nodes == nodes]
            lab = f"{nodes} CPU node{'s' if nodes > 1 else ''}"
            bars.append((f"{lab} (p={int(nd.p.max())})" if len(nd) else lab,
                         float(nd.sort_values("p").t_iter.iloc[-1]) if len(nd) else None))
        # 2 GPUs: both in one node, or one on each of two nodes (the halo crosses 100 GbE)
        for gpus, nodes, halo in ((1, 1, "aware"), (2, 1, "aware"), (2, 1, "staged"),
                                  (2, 2, "aware"), (2, 2, "staged")):
            m = grp[(grp.gpus == gpus) & (grp.nodes == nodes) & (grp.halo == halo)]
            where = "" if gpus == 1 else (", 1 node" if nodes == 1 else ", 2 nodes")
            bars.append((f"{gpus} GPU{'s' if gpus > 1 else ''}{where} ({halo})",
                         float(m.t_iter.iloc[0]) if len(m) else None))
        for j, (lab, t) in enumerate(bars):
            if t is not None:
                ax[0].bar(i + (j - (len(bars) - 1) / 2) * width, t * 1e3, width, color=f"C{j}",
                          label=lab if lab not in labelled else None)
                labelled.add(lab)
    ax[0].set_yscale("log")
    lo, hi = ax[0].get_ylim()
    ax[0].set_ylim(lo, hi * 30)  # room for the legend above the bars
    ax[0].set_xticks(range(len(names)), names)
    ax[0].set(ylabel="time per iteration [ms]", title=f"E7 CPU vs GPU ({machine})")
    ax[0].legend(fontsize=8, ncol=3, loc="upper center")
    ax[0].grid(True, axis="y", which="both", alpha=0.3)
    one = g[(g.gpus == 1) & (g.nodes == 1) & (g.halo == "aware")].sort_values("n")
    if len(one):
        ax[1].semilogx(one.n, one.gbs, "o-", label="1 GPU (achieved, 104 B/unknown model)")
        ax[1].axhline(A40_PEAK_GBS, color="k", ls="--", lw=1, label="A40 peak 696 GB/s")
        ax[1].set(xlabel="unknowns n", ylabel="effective memory bandwidth [GB/s]",
                  title="achieved bandwidth on one GPU")
        ax[1].legend(fontsize=8)
        ax[1].grid(True, which="both", alpha=0.3)
    save(fig, out, f"gpu_{machine}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="*", help="result CSV files (default results/*.csv)")
    ap.add_argument("--out", default=os.path.join("report", "figures"))
    args = ap.parse_args()
    med = median_runs(load(args.csv))
    for machine in sorted(med.machine.unique()):
        for plot in (plot_e1, plot_strong, plot_weak, plot_comm, plot_hybrid, plot_placement,
                     plot_gpu):
            plot(med, machine, args.out)


if __name__ == "__main__":
    main()

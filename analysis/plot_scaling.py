#!/usr/bin/env python3
"""Report figures from the result CSVs (experiments E1-E7 of the report).

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

# The multi-panel figures are about 12 in wide and printed at the 6.5 in text width, so the
# fonts are set large enough to stay legible after that scaling.
plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "legend.fontsize": 10})
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import amdahl_fit, load, median_runs, serial_baseline  # noqa: E402

A40_PEAK_GBS = 696.0


# Hue = dimension (2D blue, 3D orange); light with open markers = the small problem, dark and
# filled = the large one. The same problem has the same colour in every figure.
DIM_STYLE = {2: dict(color="C0", marker="o"), 3: dict(color="C1", marker="s")}
PROBLEM_STYLE = {(2, 8192): dict(color="#6baed6", marker="o", mfc="white"),
                 (2, 16384): dict(color="#08519c", marker="o"),
                 (3, 256): dict(color="#fdae6b", marker="s", mfc="white"),
                 (3, 512): dict(color="#d94801", marker="s")}


def pstyle(dim, nx):
    return PROBLEM_STYLE.get((int(dim), int(nx)), DIM_STYLE[int(dim)])


def mname(machine):
    """Machine name as written in the report (vanda -> Vanda)."""
    return machine.capitalize()


def fig_label(row):
    """Problem label for figures: 2D $8192^2$, 3D $512^3$ (2D 4096×2048 when the sides differ)."""
    d, nx, ny, nz = int(row["dim"]), int(row["nx"]), int(row["ny"]), int(row["nz"])
    if d == 2:
        return f"2D ${nx}^2$" if nx == ny else f"2D {nx}×{ny}"
    return f"3D ${nx}^3$" if nx == ny == nz else f"3D {nx}×{ny}×{nz}"


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
    for a in axes:  # the captions explain the dotted line
        a.axvline(edge, color="grey", ls=":", lw=1)


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
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for dim in (2, 3):
        st = DIM_STYLE[dim]
        f = fixed[fixed.dim == dim].sort_values("n")
        if len(f):
            ax[0].loglog(f.n, f.t_iter * 1e3, "-", **st,
                         label=f"{dim}D (slope {slope(f.n, f.t_iter):.2f})")
        t = tol[tol.dim == dim].sort_values("n")
        if len(t):
            r = t.iloc[-1]  # theory lines through the largest solved size
            ax[1].loglog(t.nx, t.iters, "-", **st,
                         label=f"{dim}D (slope {slope(t.nx, t.iters):.2f}, theory 1)")
            ax[1].loglog(t.nx, r.iters * t.nx / r.nx, "--", color=st["color"], lw=0.8)
            ax[2].loglog(t.n, t.t_total, "-", **st,
                         label=f"{dim}D (slope {slope(t.n, t.t_total):.2f}, theory {1 + 1 / dim:.2f})")
            ax[2].loglog(t.n, r.t_total * (t.n / r.n) ** (1 + 1 / dim), "--", color=st["color"], lw=0.8)
    if len(fixed):
        n = np.array([fixed.n.min(), fixed.n.max()], float)
        ref = fixed.sort_values("n").iloc[-1]
        ax[0].loglog(n, ref.t_iter * 1e3 * n / ref.n, "k--", lw=1, label="slope 1 (linear in n)")
    ax[0].set(xlabel="unknowns n", ylabel="time per CG iteration [ms]",
              title=f"E1 time per iteration ({mname(machine)})")
    ax[1].set(xlabel="grid points per dimension N", ylabel=r"CG iterations to $10^{-8}$",
              title=r"E1 iterations ($\kappa \propto N^2$)")
    ax[2].set(xlabel="unknowns n", ylabel="time to solution [s]", title="E1 time to solution")
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
        a.legend()
    plain_ticks(ax[1], sorted(tol.nx.unique())[::2] if len(tol) else None)
    save(fig, out, f"e1_{machine}")


def plot_strong(med, machine, out):
    s = med[(med.machine == machine) & (med.tag == "strong") & (med.threads == 1)]
    s = s[(s.decomp == "block") & (s.halo == "nonblocking") & (s.cg == "standard") &
          (s["dot"] == "fast")]
    if s.empty:
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for (src, dim, nx, ny, nz), grp in s.groupby(["source", "dim", "nx", "ny", "nz"]):
        grp = grp.sort_values("p")
        t1 = serial_baseline(med, machine, dim, nx, ny, nz, source=src)
        base = "serial"
        if t1 is None:
            p1 = grp[grp.p == 1]
            if p1.empty:
                continue
            t1, base = float(p1.t_iter.iloc[0]), "MPI p=1"
        label = fig_label(grp.iloc[0])
        st = pstyle(dim, nx)
        p = np.r_[1, grp.p.to_numpy(float)]  # p = 1 is T1, the serial build
        t = np.r_[t1, grp.t_iter.to_numpy(float)]
        sp = t1 / t
        ax[0].loglog(p, t * 1e3, "-", **st, label=label)
        ax[0].loglog(p, t1 * 1e3 / p, "--", color=st["color"], lw=0.8)
        ax[1].loglog(p, sp, "-", **st)
        # Amdahl's law (Lecture 7) fitted to the single-node points only, drawn over every p:
        # the multi-node points show how far the law's ceiling holds beyond one node
        one = grp[grp.nodes == 1]
        f = amdahl_fit(one.p, t1 / one.t_iter)
        if f == f and f > 0:
            pp = np.geomspace(1, grp.p.max(), 50)
            ax[1].loglog(pp, 1 / (f + (1 - f) / pp), ":", color=st["color"], lw=1.2)
        ax[2].semilogx(p, sp / p, "-", **st)
    pmax = s.p.max()
    ax[1].loglog([1, pmax], [1, pmax], "k--", lw=1, label="ideal")
    ax[1].plot([], [], ":", color="grey", lw=1.2, label="Amdahl fit to 1 node")
    ax[2].axhline(1.0, color="k", ls="--", lw=1)
    ax[0].set(xlabel="MPI ranks p", ylabel="time per iteration [ms]",
              title=f"E3 strong scaling ({mname(machine)})")
    ax[1].set(xlabel="MPI ranks p", ylabel="speedup $T_1 / T_p$", title="speedup")
    ax[2].set(xlabel="MPI ranks p", ylabel="parallel efficiency $S_p / p$", title="efficiency")
    mark_nodes(ax, s)
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
        plain_ticks(a, np.r_[1, s.p.unique()])
    ax[0].legend()
    ax[1].legend(loc="upper left")
    save(fig, out, f"strong_{machine}")


def plot_weak(med, machine, out):
    w = med[(med.machine == machine) & (med.tag == "weak") & (med.threads == 1)]
    if w.empty:
        return
    w = w.assign(local=(w.nx // w.px).astype(int))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for (dim, local), grp in w.groupby(["dim", "local"]):
        grp = grp.sort_values("p")
        ref = grp[grp.p == 1]
        if ref.empty:
            ref = grp.iloc[[0]]
        t1 = float(ref.t_iter.iloc[0])
        label = f"{dim}D, ${local}^{dim}$ per rank"
        st = DIM_STYLE[int(dim)]
        ax[0].loglog(grp.p, grp.t_iter * 1e3, "-", **st, label=label)
        ax[0].axhline(t1 * 1e3, color=st["color"], ls="--", lw=0.8)
        ax[1].semilogx(grp.p, t1 / grp.t_iter, "-", **st)
        ax[2].loglog(grp.p, grp.p * t1 / grp.t_iter, "-", **st)
    ax[1].axhline(1.0, color="k", ls="--", lw=1, label="ideal")
    ax[2].loglog([1, w.p.max()], [1, w.p.max()], "k--", lw=1, label="ideal (= p)")
    ax[2].set(xlabel="MPI ranks p", ylabel=r"scaled speedup $p\,T(1)/T(p)$",
              title="Gustafson scaled speedup")
    ax[0].set(xlabel="MPI ranks p", ylabel="time per iteration [ms]",
              title=f"E4 weak scaling ({mname(machine)})")
    ax[1].set(xlabel="MPI ranks p", ylabel="weak efficiency $T(1)/T(p)$", title="weak efficiency")
    mark_nodes(ax, w)
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
        a.legend()
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
        fig, ax = plt.subplots(figsize=(8, 0.34 * len(grp) + 1.8), layout="constrained")
        for col, lab, mk, colour in (("t_iter", "total (slowest rank)", "o", "black"),
                                     ("comp_per_iter", "stencil + vector updates", "^", "#4C72B0"),
                                     ("halo_per_iter", "halo exchange", "s", "#DD8452"),
                                     ("reduce_per_iter", "Allreduce", "D", "#55A868")):
            ax.semilogx(grp[col].to_numpy() * 1e3, y, mk, color=colour, ms=5, ls="none", label=lab)
        ax.set_yticks(y, labels, fontsize=11)
        ax.set_ylim(-0.7, len(grp) - 0.3)
        ax.set(xlabel="time per iteration [ms] (phases averaged over ranks)",
               title=f"E5 communication variants, {fig_label(grp.iloc[0])}, p = {p} on "
                     f"{nodes} node{'s' if nodes > 1 else ''} ({mname(machine)})")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.grid(True, axis="x", which="both", alpha=0.3)
        fig.legend(ncol=2, loc="outside lower center")
        save(fig, out, f"comm_{machine}_{dim}d_p{p}")


def plot_hybrid(med, machine, out):
    h = med[(med.machine == machine) & (med.tag == "hybrid")]
    if h.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for (cores, nodes), grp in h.groupby(["cores", "nodes"]):
        grp = grp.sort_values("threads")
        ax.loglog(grp.threads, grp.t_iter * 1e3, "-", **pstyle(grp.iloc[0].dim, grp.iloc[0].nx),
                  label=f"{cores} cores on {nodes} node{'s' if nodes > 1 else ''} "
                        f"({fig_label(grp.iloc[0])})")
        ys = grp.t_iter.to_numpy() * 1e3
        for k, r in enumerate(grp.itertuples()):  # ranks per node x threads
            # a label above a local minimum would sit on the line, so it goes below
            below = 0 < k < len(ys) - 1 and ys[k] < min(ys[k - 1], ys[k + 1])
            ax.annotate(f"{r.p // nodes}×{r.threads}", (r.threads, ys[k]), fontsize=10,
                        textcoords="offset points", xytext=(3, -13) if below else (3, 4))
    ax.set(xlabel="OpenMP threads per rank (ranks per node × threads = cores per node)",
           ylabel="time per iteration [ms], log scale", title=f"E6 hybrid MPI+OpenMP ({mname(machine)})")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo / 1.15, hi * 1.15)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plain_ticks(ax, h.threads.unique())
    log_yticks(ax)
    save(fig, out, f"hybrid_{machine}")


def plot_placement(med, machine, out):
    pl = med[(med.machine == machine) & med.tag.str.startswith("placement-")]
    if pl.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    style = {"core": dict(marker="o", ls="-", color="C0"),
             "numa": dict(marker="s", ls="-", color="C3", mfc="none", ms=8),
             "socket": dict(marker="^", ls="--", color="C2", mfc="none", ms=6),
             "l3cache": dict(marker="x", ls=":", color="C1", ms=5)}
    for tag, grp in pl.groupby("tag"):
        grp = grp.sort_values("p")
        name = tag.split("-", 1)[1]
        ax.loglog(grp.p, grp.t_iter * 1e3, label=name, **style.get(name, dict(marker="o")))
    r = pl.iloc[0]
    t1 = serial_baseline(med, machine, r.dim, r.nx, r.ny, r.nz)
    if t1:
        pp = np.array([pl.p.min(), pl.p.max()], float)
        ax.loglog(pp, t1 * 1e3 / pp, "k--", lw=1, label="ideal $T_1/p$")
    full = med[(med.machine == machine) & (med.tag == "strong") & (med.dim == r.dim) &
               (med.nx == r.nx) & (med.threads == 1) & (med.nodes == 1)]
    if len(full):  # one node only: this figure is about placement within a node
        best = full.sort_values("p").iloc[-1]
        ax.axhline(best.t_iter * 1e3, color="0.4", ls=":", lw=1,
                   label=f"all {int(best.p)} cores (memory-bandwidth limit)")
    ax.set(xlabel="MPI ranks p on one node", ylabel="time per iteration [ms]",
           title=f"Rank placement within one node, {fig_label(pl.iloc[0])} ({mname(machine)})")
    plain_ticks(ax, pl.p.unique())
    log_yticks(ax)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="mpirun --map-by", loc="lower left", ncol=2)
    save(fig, out, f"placement_{machine}")


def plot_gpu(med, machine, out):
    g = med[(med.machine == machine) & (med.tag == "gpu-strong")]
    if g.empty:
        return
    problems = g.groupby(["dim", "nx", "ny", "nz"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    names = []
    labelled = set()
    colour = {}
    for i, ((dim, nx, ny, nz), grp) in enumerate(problems):
        names.append(fig_label(grp.iloc[0]))
        bars = []
        t1 = serial_baseline(med, machine, dim, nx, ny, nz)
        if t1:
            bars.append(("1 CPU core (serial)", t1))
        cpu = med[(med.machine == machine) & (med.tag == "strong") & (med.dim == dim) &
                  (med.nx == nx) & (med.ny == ny) & (med.nz == nz) & (med.threads == 1) &
                  (med.decomp == "block") & (med.halo == "nonblocking") & (med.cg == "standard") &
                  (med["dot"] == "fast")]
        for nodes in sorted(int(k) for k in cpu.nodes.unique()):  # every node count measured
            nd = cpu[cpu.nodes == nodes].sort_values("p")
            bars.append((f"{nodes} CPU node{'s' if nodes > 1 else ''} (p = {int(nd.p.iloc[-1])})",
                         float(nd.t_iter.iloc[-1])))
        # 2 GPUs: both in one node, or one on each of two nodes (the halo crosses 100 GbE)
        for gpus, nodes, halo in ((1, 1, "aware"), (2, 1, "aware"), (2, 1, "staged"),
                                  (2, 2, "aware"), (2, 2, "staged")):
            m = grp[(grp.gpus == gpus) & (grp.nodes == nodes) & (grp.halo == halo)]
            where = "" if gpus == 1 else (" in 1 node" if nodes == 1 else " on 2 nodes")
            mode = "" if gpus == 1 else f", {'CUDA-aware' if halo == 'aware' else 'host-staged'}"
            if len(m):
                bars.append((f"{gpus} A40{where}{mode}", float(m.t_iter.iloc[0])))
        width = 0.8 / len(bars)
        xs = {lab: i + (j - (len(bars) - 1) / 2) * width for j, (lab, _) in enumerate(bars)}
        for lab, t in bars:
            c = colour.setdefault(lab.split(" (")[0], f"C{len(colour)}")
            ax[0].bar(xs[lab], t * 1e3, width, color=c, label=lab if lab not in labelled else None)
            labelled.add(lab)
        gpu1 = grp[(grp.gpus == 1) & (grp.halo == "aware")]
        two = [xs[lab] for lab, _ in bars if lab.startswith("2 A40")]
        if len(gpu1) and two:  # ideal two-GPU time: half the one-GPU time
            ax[0].hlines(float(gpu1.t_iter.iloc[0]) * 1e3 / 2, min(two) - width / 2, max(two) + width / 2,
                         color="k", lw=1.2, ls="--",
                         label="ideal 2 A40 (1 A40 / 2)" if "ideal" not in labelled else None)
            labelled.add("ideal")
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(len(names)), names)
    ax[0].set(ylabel="time per iteration [ms]", title=f"E7 CPU against GPU ({mname(machine)})")
    handles, labels = ax[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda k: labels[k].startswith("ideal"))  # ideal last
    # three columns: CPU bars, then the GPU bars, then the ideal line
    fig.legend([handles[k] for k in order], [labels[k] for k in order], ncol=3, loc="outside lower center")
    ax[0].grid(True, axis="y", which="both", alpha=0.3)
    one = g[(g.gpus == 1) & (g.nodes == 1) & (g.halo == "aware")].sort_values("n")
    if len(one):
        ca, cn = colour.get("1 A40", "C2"), colour.get("1 CPU node", "C1")
        for dim, sub in one.groupby("dim"):
            ax[1].semilogx(sub.n, sub.gbs, "o-" if dim == 2 else "s-", color=ca,
                           label=f"1 A40, {dim}D")
        node = med[(med.machine == machine) & (med.tag == "strong") & (med.threads == 1) &
                   (med.nodes == 1) & (med.p == 72) & (med.decomp == "block") &
                   (med.halo == "nonblocking") & (med.cg == "standard") & (med["dot"] == "fast") &
                   med.n.isin(one.n)].sort_values("n")
        for dim, sub in node.groupby("dim"):
            ax[1].semilogx(sub.n, sub.gbs, "o--" if dim == 2 else "s--", color=cn,
                           label=f"1 CPU node (p = 72), {dim}D")
        ax[1].axhline(A40_PEAK_GBS, color=ca, ls=":", lw=1.2, label="A40 peak 696 GB/s")
        ax[1].axhline(614, color=cn, ls=":", lw=1.2, label="CPU node peak 614 GB/s")
        ax[1].set_ylim(0, 760)  # from 0: the panel compares achieved with peak bandwidth
        ax[1].set(xlabel="unknowns n", ylabel="achieved bandwidth [GB/s]",
                  title="bandwidth (104 bytes per unknown)")
        ax[1].set_xticks([2e7, 5e7, 1e8, 2e8], [r"$2\cdot10^7$", r"$5\cdot10^7$", r"$10^8$", r"$2\cdot10^8$"])
        ax[1].xaxis.set_minor_formatter(NullFormatter())
        ax[1].legend(loc="lower center", ncol=2, fontsize=9)
        ax[1].grid(True, which="both", alpha=0.3)
    save(fig, out, f"gpu_{machine}")


def plot_decomposition(out):
    """Schematic, no data: (a) a 2D grid on 2 x 2 ranks with one rank's ghost cells and the
    5-point stencil next to a box face; (b)-(d) slab, pencil and block boxes on 8 ranks."""
    from matplotlib.patches import Rectangle
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(8.6, 2.9))
    ax = fig.add_subplot(1, 4, 1)
    n, h = 8, 4                                   # 8 x 8 interior points, 4 x 4 per rank
    for i in range(n):
        for j in range(n):
            ax.plot(i, j, "o", ms=3.5, color="C0" if (i >= h and j >= h) else "0.6")
    for i, j in [(h - 1, j) for j in range(h, n)] + [(i, h - 1) for i in range(h, n)]:
        ax.add_patch(Rectangle((i - .4, j - .4), .8, .8, fill=False, hatch="////", ec="C1", lw=1))
    for k in range(h, n):                         # physical boundary: the ghost value is 0
        ax.plot(n, k, "s", ms=3.5, mfc="white", mec="0.3")
        ax.plot(k, n, "s", ms=3.5, mfc="white", mec="0.3")
    ax.plot([h - .5, h - .5], [-.5, n - .5], "k", lw=1.5)
    ax.plot([-.5, n - .5], [h - .5, h - .5], "k", lw=1.5)
    ax.add_patch(Rectangle((-.5, -.5), n, n, fill=False, ec="k", lw=1.5))
    si, sj = h, h + 2                             # the 5-point stencil next to the box's left face
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ax.plot([si, si + di], [sj, sj + dj], color="C3", lw=2)
    ax.plot(si, sj, "o", ms=8, color="C3")
    for (x, y), r in zip(((1.5, 1.5), (5.5, 1.5), (1.5, 5.5), (5.5, 5.5)), (0, 2, 1, 3)):
        ax.text(x, y, f"rank {r}", ha="center", va="center", fontsize=9,
                bbox=dict(fc="white", ec="none", alpha=.8))
    ax.set_xlim(-1, n + .7)
    ax.set_ylim(-1, n + .7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) 2D grid on 2 × 2 ranks", fontsize=9)

    def boxes(a, dims, title):
        px, py, pz = dims
        for ix in range(px):
            for iy in range(py):
                for iz in range(pz):
                    x0, y0, z0 = ix / px, iy / py, iz / pz
                    x1, y1, z1 = x0 + 1 / px, y0 + 1 / py, z0 + 1 / pz
                    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
                    faces = [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
                             [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
                             [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]]]
                    one = (ix, iy, iz) == (px - 1, 0, pz - 1)   # one rank's box, shaded
                    a.add_collection3d(Poly3DCollection(faces, facecolor="C0" if one else "0.92",
                                                        edgecolor="k", lw=.6, alpha=.9 if one else .25))
        a.set_title(title, fontsize=9)
        a.set_box_aspect((1, 1, 1))
        a.view_init(elev=22, azim=-58)
        a.set_axis_off()

    for k, (dims, t) in enumerate((((8, 1, 1), "(b) slab, 8 × 1 × 1 ranks"),
                                   ((4, 2, 1), "(c) pencil, 4 × 2 × 1 ranks"),
                                   ((2, 2, 2), "(d) block, 2 × 2 × 2 ranks"))):
        boxes(fig.add_subplot(1, 4, k + 2, projection="3d"), dims, t)
    save(fig, out, "decomposition")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="*", help="result CSV files (default results/*.csv)")
    ap.add_argument("--out", default=os.path.join("report", "figures"))
    args = ap.parse_args()
    med = median_runs(load(args.csv))
    plot_decomposition(args.out)
    for machine in sorted(med.machine.unique()):
        for plot in (plot_e1, plot_strong, plot_weak, plot_comm, plot_hybrid, plot_placement,
                     plot_gpu):
            plot(med, machine, args.out)


if __name__ == "__main__":
    main()

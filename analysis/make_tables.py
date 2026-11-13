#!/usr/bin/env python3
"""Report tables (Markdown) from the result CSVs: serial scaling, strong-scaling speedup and
efficiency, weak-scaling efficiency, the communication breakdown, the multi-node transport
control and the GPU runs.

    python3 analysis/make_tables.py [CSV ...] [--out report/tables]

Speedups use the serial build on the same machine as T1 (MPI p = 1 if no serial run exists),
following PROJECT_PLAN.md §4.4. Medians over repeated runs.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import amdahl_fit, karp_flatt, load, median_runs, serial_baseline, size_label  # noqa: E402


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def e1_table(med, machine):
    f = med[(med.machine == machine) & (med.backend == "serial") & (med.tag == "e1")]
    t = med[(med.machine == machine) & (med.backend == "serial") & (med.tag == "e1-tol")]
    rows = []
    for r in f.sort_values(["dim", "n"]).itertuples():
        tt = t[(t.dim == r.dim) & (t.nx == r.nx)]
        iters = int(tt.iters.iloc[0]) if len(tt) else ""
        tsol = f"{tt.t_total.iloc[0]:.3f}" if len(tt) else ""
        rows.append([f"{r.dim}D", r.nx, f"{r.n:,}", f"{r.nnz:,}", f"{r.t_iter * 1e3:.3f}",
                     f"{r.gbs:.1f}", iters, tsol])
    return md_table(["dim", "N", "n", "nnz", "ms / iteration", "GB/s (model)",
                     "iterations to 1e-8", "time to 1e-8 [s]"], rows) if rows else ""


def strong_table(med, machine):
    s = med[(med.machine == machine) & (med.tag == "strong") & (med.threads == 1) &
            (med.decomp == "block") & (med.halo == "nonblocking") & (med.cg == "standard") &
            (med["dot"] == "fast")]
    out = ""
    for (src, dim, nx, ny, nz), grp in s.groupby(["source", "dim", "nx", "ny", "nz"]):
        t1 = serial_baseline(med, machine, dim, nx, ny, nz, source=src)
        base = "serial"
        if t1 is None and (grp.p == 1).any():
            t1, base = float(grp[grp.p == 1].t_iter.iloc[0]), "MPI p=1"
        if t1 is None:
            continue
        rows = []
        grp = grp.sort_values("p")
        speedups = t1 / grp.t_iter
        for r, sp in zip(grp.itertuples(), speedups):
            e = karp_flatt(sp, r.p)
            rows.append([r.p, r.nodes, f"{r.px}x{r.py}x{r.pz}", f"{r.t_iter * 1e3:.3f}", f"{sp:.2f}",
                         f"{sp / r.p:.2f}", "–" if r.p == 1 else f"{e:.3f}",
                         f"{r.halo_per_iter * 1e3:.3f}", f"{r.reduce_per_iter * 1e3:.3f}"])
        f = amdahl_fit(grp.p, speedups)
        out += (f"\n**Strong scaling, {size_label(grp.iloc[0])}, {machine}** "
                f"(T1 = {t1 * 1e3:.3f} ms/iteration, {base}; Amdahl fit: serial fraction "
                f"f = {f:.3f}, so S(∞) = {1 / f if f > 0 else float('inf'):.0f})\n\n")
        out += md_table(["p", "nodes", "ranks", "ms / iteration", "speedup", "efficiency",
                         "Karp–Flatt e", "halo ms", "Allreduce ms"], rows)
    return out


def weak_table(med, machine):
    w = med[(med.machine == machine) & (med.tag == "weak") & (med.threads == 1)]
    out = ""
    if w.empty:
        return out
    w = w.assign(local=(w.nx // w.px).astype(int))
    for (dim, local), grp in w.groupby(["dim", "local"]):
        grp = grp.sort_values("p")
        t1 = float(grp.t_iter.iloc[0])
        rows = []
        for r in grp.itertuples():
            eff = t1 / r.t_iter
            scaled = r.p * eff  # Gustafson's scaled speedup
            alpha = (r.p - scaled) / (r.p - 1) if r.p > 1 else float("nan")
            rows.append([r.p, r.nodes, size_label(r._asdict()), f"{r.t_iter * 1e3:.3f}", f"{eff:.2f}",
                         f"{scaled:.1f}", "–" if r.p == 1 else f"{alpha:.3f}"])
        out += f"\n**Weak scaling, {local}^{dim} unknowns per rank, {machine}**\n\n"
        out += md_table(["p", "nodes", "global grid", "ms / iteration", "weak efficiency",
                         "scaled speedup (Gustafson)", "implied serial fraction"], rows)
    return out


def comm_table(med, machine):
    c = med[(med.machine == machine) & (med.tag == "comm")]
    out = ""
    for (p, dim, nx), grp in c.groupby(["p", "dim", "nx"]):
        rows = [[r.decomp, r.halo, r.cg, r.dot, f"{r.t_iter * 1e3:.3f}", f"{r.comp_per_iter * 1e3:.3f}",
                 f"{r.halo_per_iter * 1e3:.3f}", f"{r.reduce_per_iter * 1e3:.3f}"]
                for r in grp.sort_values("t_iter").itertuples()]
        out += f"\n**Communication variants, {size_label(grp.iloc[0])}, p = {p}, {machine}**\n\n"
        out += md_table(["decomposition", "halo", "CG", "dot products", "ms / iteration", "compute ms",
                         "halo ms", "Allreduce ms"], rows)
    return out


def control_table(med, machine):
    """Multi-node transport on one node (tag "control", p = 64 on the job's first node) against
    the single-node job's p = 64 run with the module's default transport (ob1 + shared memory)."""
    c = med[(med.machine == machine) & (med.tag == "control")]
    if c.empty:
        return ""
    ref = med[(med.machine == machine) & (med.tag == "strong") & (med.threads == 1) &
              (med.nodes == 1) & (med.decomp == "block") & (med.halo == "nonblocking") &
              (med.cg == "standard") & (med["dot"] == "fast")]
    rows = []
    for r in c.sort_values(["dim", "nx", "nodes"]).itertuples():
        m = ref[(ref.p == r.p) & (ref.dim == r.dim) & (ref.nx == r.nx) & (ref.ny == r.ny) &
                (ref.nz == r.nz)]
        t0 = float(m.t_iter.iloc[0]) if len(m) else None
        rows.append([size_label(r._asdict()), r.p, f"{r.nodes}-node job",
                     f"{t0 * 1e3:.3f}" if t0 else "–", f"{r.t_iter * 1e3:.3f}",
                     f"{r.t_iter / t0:.3f}" if t0 else "–", f"{r.halo_per_iter * 1e3:.3f}",
                     f"{r.reduce_per_iter * 1e3:.3f}"])
    return (f"\n**Transport control, {machine}:** p = 64 on one node with the multi-node transport "
            "(UCX over TCP) versus the single-node job (module default: ob1 + shared memory)\n\n" +
            md_table(["problem", "p", "run inside", "ob1 ms / iteration", "UCX ms / iteration",
                      "ratio", "UCX halo ms", "UCX Allreduce ms"], rows))


def gpu_table(med, machine):
    g = med[(med.machine == machine) & (med.tag == "gpu-strong")]
    out = ""
    for (dim, nx, ny, nz), grp in g.groupby(["dim", "nx", "ny", "nz"]):
        one = grp[(grp.gpus == 1) & (grp.halo == "aware")]
        t1 = float(one.t_iter.iloc[0]) if len(one) else None
        rows = []
        for r in grp.sort_values(["gpus", "nodes", "halo"]).itertuples():
            rows.append([r.gpus, r.nodes, r.halo, f"{r.t_iter * 1e3:.3f}",
                         f"{t1 / r.t_iter:.2f}" if t1 else "–", f"{r.comp_per_iter * 1e3:.3f}",
                         f"{r.halo_per_iter * 1e3:.3f}", f"{r.reduce_per_iter * 1e3:.3f}",
                         f"{r.gbs:.0f}"])
        out += f"\n**GPU runs (E7), {size_label(grp.iloc[0])}, {machine}**\n\n"
        out += md_table(["GPUs", "nodes", "halo exchange", "ms / iteration", "speedup vs 1 GPU",
                         "compute ms", "halo ms", "Allreduce ms", "GB/s (model)"], rows)
    return out


# ---- Compact tables for report/report.md (inlined by report/build.py) --------------------------

def sup(n):
    return str(n).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


def short_label(dim, nx, ny, nz):
    if dim == 2:
        return f"2D {nx}²" if nx == ny else f"2D {nx}×{ny}"
    return f"3D {nx}³" if nx == ny == nz else f"3D {nx}×{ny}×{nz}"


def strong_series(med, machine):
    """{label: (t1, base, DataFrame sorted by p)} for the strong-scaling problems."""
    s = med[(med.machine == machine) & (med.tag == "strong") & (med.threads == 1) &
            (med.decomp == "block") & (med.halo == "nonblocking") & (med.cg == "standard") &
            (med["dot"] == "fast")]
    out = {}
    for (src, dim, nx, ny, nz), grp in sorted(s.groupby(["source", "dim", "nx", "ny", "nz"]),
                                              key=lambda kv: (kv[0][1], kv[0][2])):
        t1 = serial_baseline(med, machine, dim, nx, ny, nz, source=src)
        base = "serial"
        if t1 is None and (grp.p == 1).any():
            t1, base = float(grp[grp.p == 1].t_iter.iloc[0]), "MPI p = 1"
        if t1 is not None:
            out[short_label(dim, nx, ny, nz)] = (t1, base, grp.sort_values(["p", "nodes"]))
    return out


def report_strong(med, machine):
    """Speedup / efficiency for every problem in one table, and the Karp-Flatt serial fraction."""
    series = strong_series(med, machine)
    if not series:
        return "", ""
    keys = sorted({(int(r.p), int(r.nodes)) for _, _, g in series.values() for r in g.itertuples()})
    rows, kf = [], []
    for p, nodes in keys:
        row, krow = [p, nodes], [p, nodes]
        for t1, _, g in series.values():
            m = g[(g.p == p) & (g.nodes == nodes)]
            if len(m):
                sp = t1 / float(m.t_iter.iloc[0])
                row.append(f"{sp:.1f} / {sp / p:.2f}")
                krow.append("–" if p == 1 else f"{karp_flatt(sp, p):.3f}")
            else:
                row.append("")
                krow.append("")
        rows.append(row)
        kf.append(krow)
    head = ["p", "nodes"] + list(series)
    return md_table(head, rows), md_table(head, kf)


def report_strong_baselines(med, machine):
    """One sentence: T1 and the Amdahl fit of every strong-scaling problem."""
    parts = []
    for label, (t1, base, g) in strong_series(med, machine).items():
        f = amdahl_fit(g.p, t1 / g.t_iter)
        parts.append(f"{label}: $T_1$ = {t1 * 1e3:.1f} ms, Amdahl fit $f$ = {f:.3f}")
    return ("**Serial baselines and Amdahl fits.** " + "; ".join(parts) + ".") if parts else ""


def report_weak(med, machine):
    w = med[(med.machine == machine) & (med.tag == "weak") & (med.threads == 1)]
    if w.empty:
        return ""
    w = w.assign(local=(w.nx // w.px).astype(int))
    groups = {}
    for (dim, local), grp in w.groupby(["dim", "local"]):
        grp = grp.sort_values(["p", "nodes"])
        groups[f"{dim}D, {local}{sup(dim)} per rank"] = (float(grp.t_iter.iloc[0]), grp)
    keys = sorted({(int(r.p), int(r.nodes)) for _, g in groups.values() for r in g.itertuples()})
    rows = []
    for p, nodes in keys:
        row = [p, nodes]
        for t1, g in groups.values():
            m = g[(g.p == p) & (g.nodes == nodes)]
            if len(m):
                t = float(m.t_iter.iloc[0])
                row += [f"{t * 1e3:.1f}", f"{t1 / t:.2f}", f"{p * t1 / t:.1f}"]
            else:
                row += ["", "", ""]
        rows.append(row)
    head = ["p", "nodes"]
    for label in groups:
        head += [f"{label}: ms", "E_w", "scaled S"]
    return md_table(head, rows)


COMM_ROWS = [("sendrecv", "standard", "fast", "sendrecv"), ("nonblocking", "standard", "fast", "nonblocking"),
             ("overlap", "standard", "fast", "overlap"), ("rma", "standard", "fast", "RMA, fence"),
             ("pscw", "standard", "fast", "RMA, PSCW"),
             ("nonblocking", "fused", "fast", "nonblocking + fused CG"),
             ("nonblocking", "standard", "repro", "nonblocking + repro dots"),
             ("nonblocking", "fused", "repro", "nonblocking + fused CG + repro dots")]


def report_comm(med, machine):
    """{(p, dim, nx, nodes): table}: rows = halo / CG / dot variant, columns = decomposition;
    each cell 'total ms (halo ms)'."""
    c = med[(med.machine == machine) & (med.tag == "comm")]
    out = {}
    for (p, dim, nx, nodes), grp in c.groupby(["p", "dim", "nx", "nodes"]):
        decomps = [d for d in ("block", "pencil", "slab") if (grp.decomp == d).any()]
        rows = []
        for halo, cg, dot, name in COMM_ROWS:
            cells = []
            for d in decomps:
                m = grp[(grp.decomp == d) & (grp.halo == halo) & (grp.cg == cg) & (grp["dot"] == dot)]
                cells.append(f"{m.t_iter.iloc[0] * 1e3:.1f} ({m.halo_per_iter.iloc[0] * 1e3:.2f})"
                             if len(m) else "")
            if any(cells):
                rows.append([name] + cells)
        out[(int(p), int(dim), int(nx), int(nodes))] = md_table(["variant"] + decomps, rows)
    return out


def report_gpu(med, machine):
    g = med[(med.machine == machine) & (med.tag == "gpu-strong")]
    if g.empty:
        return ""
    probs = sorted({(int(r.dim), int(r.nx), int(r.ny), int(r.nz)) for r in g.itertuples()})
    cpu = med[(med.machine == machine) & (med.tag == "strong") & (med.threads == 1) &
              (med.decomp == "block") & (med.halo == "nonblocking") & (med.cg == "standard") &
              (med["dot"] == "fast")]
    configs = [("1 CPU core (serial)", lambda d, x, y, z: serial_baseline(med, machine, d, x, y, z))]
    for nodes in sorted(cpu.nodes.unique()):
        sub = cpu[cpu.nodes == nodes]
        pmax = int(sub.p.max())

        def f(d, x, y, z, sub=sub, pmax=pmax):
            m = sub[(sub.dim == d) & (sub.nx == x) & (sub.ny == y) & (sub.nz == z) & (sub.p == pmax)]
            return float(m.t_iter.iloc[0]) if len(m) else None
        configs.append((f"{nodes} CPU node{'s' if nodes > 1 else ''} (p = {pmax})", f))
    for gpus, nodes, halo in ((1, 1, "aware"), (2, 1, "aware"), (2, 1, "staged"), (2, 2, "aware"),
                              (2, 2, "staged")):
        def f(d, x, y, z, gpus=gpus, nodes=nodes, halo=halo):
            m = g[(g.gpus == gpus) & (g.nodes == nodes) & (g.halo == halo) & (g.dim == d) &
                  (g.nx == x) & (g.ny == y) & (g.nz == z)]
            return float(m.t_iter.iloc[0]) if len(m) else None
        where = "" if gpus == 1 else (" in 1 node" if nodes == 1 else " on 2 nodes")
        configs.append((f"{gpus} A40{where}, {'CUDA-aware' if halo == 'aware' else 'host-staged'}", f))
    rows = []
    for name, f in configs:
        vals = [f(*pr) for pr in probs]
        if any(v is not None for v in vals):
            rows.append([name] + [("" if v is None else f"{v * 1e3:.2f}" if v < 0.01 else f"{v * 1e3:.1f}") for v in vals])
    return md_table(["configuration"] + [short_label(*pr) for pr in probs], rows)


def report_hybrid(med, machine):
    h = med[(med.machine == machine) & (med.tag == "hybrid")]
    if h.empty:
        return ""
    cols = sorted({(int(r.nodes), int(r.cores)) for r in h.itertuples()})
    threads = sorted(h.threads.unique())
    rows = []
    for t in threads:
        row, label = [], None
        for nodes, cores in cols:
            m = h[(h.threads == t) & (h.nodes == nodes) & (h.cores == cores)]
            if len(m):
                label = label or f"{int(m.p.iloc[0]) // nodes} × {t}"
                row.append(f"{m.t_iter.iloc[0] * 1e3:.1f}")
            else:
                row.append("")
        rows.append([label or f"? × {t}"] + row)
    return md_table(["ranks × threads per node"] +
                    [f"{nodes} node{'s' if nodes > 1 else ''} ({cores} cores): ms" for nodes, cores in cols], rows)


def write_report_tables(med, machine, outdir):
    files = {f"report_e1_{machine}.md": e1_table(med, machine)}
    st, kf = report_strong(med, machine)
    files[f"report_strong_{machine}.md"] = st
    files[f"report_karpflatt_{machine}.md"] = kf
    files[f"report_strong_baselines_{machine}.md"] = report_strong_baselines(med, machine) + "\n"
    files[f"report_weak_{machine}.md"] = report_weak(med, machine)
    for (p, dim, nx, nodes), table in report_comm(med, machine).items():
        files[f"report_comm_{machine}_{dim}d_p{p}.md"] = table
    files[f"report_gpu_{machine}.md"] = report_gpu(med, machine)
    files[f"report_hybrid_{machine}.md"] = report_hybrid(med, machine)
    files[f"report_control_{machine}.md"] = control_table(med, machine).split("\n\n", 1)[-1].lstrip() \
        if not med[(med.machine == machine) & (med.tag == "control")].empty else ""
    for name, text in files.items():
        if text:
            with open(os.path.join(outdir, name), "w") as fh:
                fh.write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="*", help="result CSV files (default results/*.csv)")
    ap.add_argument("--out", default=os.path.join("report", "tables"))
    args = ap.parse_args()
    med = median_runs(load(args.csv))
    os.makedirs(args.out, exist_ok=True)
    for machine in sorted(med.machine.unique()):
        text = f"# Tables for {machine}\n\n"
        e1 = e1_table(med, machine)
        if e1:
            text += f"**Serial scaling (E1), {machine}**\n\n{e1}"
        text += (strong_table(med, machine) + weak_table(med, machine) + comm_table(med, machine) +
                 control_table(med, machine) + gpu_table(med, machine))
        path = os.path.join(args.out, f"tables_{machine}.md")
        with open(path, "w") as fh:
            fh.write(text)
        print(f"wrote {path}")
        write_report_tables(med, machine, args.out)


if __name__ == "__main__":
    main()

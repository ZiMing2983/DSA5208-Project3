"""Loading and summarising the result CSVs written by poisson_serial / poisson_mpi / poisson_cuda."""
import glob
import os
import re

import pandas as pd

# Minimum memory traffic of one CG iteration per unknown: 13 doubles (stencil 2, p.q 2,
# x/r update 6, p update 3), for both CG variants. Used for the achieved-bandwidth estimates.
BYTES_PER_UNKNOWN = 13 * 8

CONFIG = ["machine", "backend", "tag", "dim", "nx", "ny", "nz", "n", "nnz", "p", "threads",
          "gpus", "decomp", "halo", "cg", "dot", "px", "py", "pz", "tol", "source", "nodes"]
TIMES = ["t_iter", "t_total", "t_comp", "t_halo", "t_reduce", "iters", "err_inf", "rel_res",
         "t_h2d", "t_d2h"]


def load(paths=None):
    """All rows of the given CSV files (default results/*.csv), with a 'tag' column."""
    if not paths:
        paths = sorted(glob.glob(os.path.join("results", "*.csv")))
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        # Files written before a column existed get its default value.
        for col, default in (("tag", ""), ("dot", "fast"), ("t_h2d", 0.0), ("t_d2h", 0.0)):
            if col not in df.columns:
                df[col] = default
        # "source" is the campaign: cpu_vanda+2node.csv belongs with cpu_vanda.csv (the
        # multi-node jobs write their own files but share the 1-node job's T1 baselines).
        df["file"] = os.path.basename(path)
        df["source"] = re.sub(r"\+[^.]*", "", os.path.basename(path))
        # Nodes the job used, from the file name (cpu_vanda+4node.csv -> 4; otherwise 1). The
        # rows themselves do not say: 2 GPUs in one node and 1 GPU on each of 2 nodes both
        # have p = gpus = 2.
        m = re.search(r"\+(\d+)node", os.path.basename(path))
        df["nodes"] = int(m.group(1)) if m else 1
        frames.append(df)
    if not frames:
        raise SystemExit("no CSV files found (looked for results/*.csv)")
    df = pd.concat(frames, ignore_index=True)
    df["tag"] = df["tag"].fillna("").astype(str)
    return df


def median_runs(df):
    """One row per configuration: the median over repeated runs, plus per-iteration phases."""
    if df.empty:
        return df
    g = df.groupby(CONFIG, dropna=False)
    out = g[TIMES].median().reset_index()
    out["reps"] = g.size().values
    for phase in ("comp", "halo", "reduce"):
        out[f"{phase}_per_iter"] = out[f"t_{phase}"] / out["iters"]
    out["cores"] = out["p"] * out["threads"]
    out["gbs"] = BYTES_PER_UNKNOWN * out["n"] / out["t_iter"] / 1e9
    return out


def karp_flatt(speedup, p):
    """Experimentally determined serial fraction e = (1/S - 1/p) / (1 - 1/p) (Lecture 7's
    Amdahl's law solved for the serial fraction S); undefined for p = 1."""
    return (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p) if p > 1 else float("nan")


def amdahl_fit(ps, speedups):
    """Least-squares serial fraction f of Amdahl's law S(p) = 1 / (f + (1 - f)/p), fitted on
    1/S - 1/p = f (1 - 1/p) over the points with p > 1."""
    num = den = 0.0
    for p, s in zip(ps, speedups):
        if p > 1:
            a = 1.0 - 1.0 / p
            num += (1.0 / s - 1.0 / p) * a
            den += a * a
    return num / den if den > 0 else float("nan")


def size_label(row):
    if int(row["dim"]) == 2:
        return f"2D {int(row['nx'])}x{int(row['ny'])}"
    return f"3D {int(row['nx'])}x{int(row['ny'])}x{int(row['nz'])}"


def serial_baseline(med, machine, dim, nx, ny, nz, source=None):
    """Serial time per iteration of this problem on this machine (T1), or None.

    With `source`, only runs from the same CSV file count, i.e. from the same job on the same
    kind of node: a serial run on a node shared with other jobs is slower (memory bandwidth
    is shared) and would inflate every speedup."""
    m = med[(med.machine == machine) & (med.backend == "serial") & (med.dim == dim) &
            (med.nx == nx) & (med.ny == ny) & (med.nz == nz) & (med.tol == 0)]
    if source is not None:
        m = m[m.source == source]
    return float(m["t_iter"].median()) if len(m) else None

#!/usr/bin/env python3
"""
zkd_colab.py  --  everything needed to finish the ZKD paper, for Google Colab
=============================================================================

Runs every measurement behind Table 1 and Section 5 of

  "Mathematical Decoration Is Not Security: A Critical Analysis of a
   Zeta-Inspired Key-Derivation Prototype"

using a pool of 2 worker processes (Colab's standard CPU runtime has 2 vCPUs).

It writes two files next to the script:
  results_colab.json   all raw numbers + environment (cite this as the artifact)
  results_macros.tex   LaTeX \\newcommand definitions; the paper \\input's it, so
                       you never copy numbers by hand.

Colab usage
-----------
  1. Runtime > Change runtime type > CPU (the default is fine).
  2. Upload this file (left sidebar > Files), or in a cell:  %%writefile zkd_colab.py
  3. In a cell:        !python zkd_colab.py
     Smoke test first: !python zkd_colab.py --quick        (about 1 minute)
  4. Download results_colab.json and results_macros.tex, put them next to
     the .tex file, recompile.

Determinism
-----------
All random inputs are drawn in the main process from one seeded generator, in a
fixed order, and the work is split into fixed-size chunks, so results do not
depend on the number of workers. Only the throughput numbers are timing
measurements. The cross-precision experiment depends on the platform libm by
design: it is the thing being measured.

What is new relative to zkd_experiments.py
------------------------------------------
* 2-worker parallelism and a measured 1-vs-2-worker speedup.
* An attack-coverage study (Section "Attack cost"): how many hash evaluations a
  *naive grid* search needs, versus the ideal "enumerate distinct keys" count.
  The ideal count is a lower bound; a grid attacker pays for the cells it hits
  more than once and misses some cells.
* Environment capture and LaTeX macro output.
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time

import numpy as np

N_BYTES = 128
T0_EXAMPLE = 84930211.5839201
ULP = 2.0 ** -26            # spacing of float64 on [2^26, 2^27) = [6.7e7, 1.34e8)
SECONDS_PER_YEAR = 31_557_600.0


# --------------------------------------------------------------------------
# ZKD stages 1-2 (vectorised)
# --------------------------------------------------------------------------
def zkd_bytes(t0, dtype=np.float64, n=N_BYTES):
    t = np.array(t0, dtype=dtype, copy=True)
    two_pi = (dtype(2) * dtype(np.pi)) if dtype is np.longdouble else 2.0 * math.pi
    out = np.empty((t.shape[0], n), dtype=np.uint8)
    for i in range(n):
        rho = np.log(t + dtype(2)) / two_pi
        t = t + dtype(1) / (rho + dtype(0.1))
        w = np.sin(t * np.log(t)) * np.cos(t / dtype(2))
        b = np.floor(dtype(127.5) * (w + dtype(1)))
        out[:, i] = (b.astype(np.int64) % 256).astype(np.uint8)
    return out


def digests(byte_rows, size=32):
    sha = hashlib.sha256
    return [sha(r.tobytes()).digest()[:size] for r in byte_rows]


# --------------------------------------------------------------------------
# Worker functions (module level so they can be pickled)
# --------------------------------------------------------------------------
def w_digests(args):
    t0, size = args
    return digests(zkd_bytes(t0), size)


def w_count_matches(args):
    """Derive keys and count how many equal `target` (minimal IPC: throughput)."""
    t0, target = args
    return sum(1 for d in digests(zkd_bytes(t0)) if d == target)


def w_bincount(t0):
    return np.bincount(zkd_bytes(t0).ravel(), minlength=256)


def w_precision(t0):
    b64 = zkd_bytes(t0, np.float64)
    b80 = zkd_bytes(t0.astype(np.longdouble), np.longdouble)
    k64, k80 = digests(b64), digests(b80)
    mism = np.array([a != b for a, b in zip(k64, k80)])
    return mism, (b64 != b80).sum(axis=1)


# --------------------------------------------------------------------------
def chunks(a, size):
    return [a[i:i + size] for i in range(0, len(a), size)]


def par_digests(pool, t0, size=32, chunk=20_000):
    out = []
    for part in pool.map(w_digests, [(c, size) for c in chunks(t0, chunk)]):
        out.extend(part)
    return out


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def exp_density_check():
    proto = lambda t: 1.0 / (math.log(t + 2) / (2 * math.pi) + 0.1)
    true = lambda t: 2 * math.pi / math.log(t / (2 * math.pi))
    return {str(t): {"prototype": proto(t), "true_mean": true(t)}
            for t in (T0_EXAMPLE, 14.134725)}


def exp_grid(pool, step, npts):
    t0 = T0_EXAMPLE + np.arange(npts, dtype=np.float64) * step
    return {"points": npts, "step": step, "width": step * npts,
            "distinct": len(set(par_digests(pool, t0)))}


def exp_perturbation(pool, rng, n):
    ladder = [f * 10.0 ** e for e in range(-8, -2) for f in (1.0, 2.0, 5.0)]
    ladder = [d for d in ladder if d <= 1e-3]
    h = rng.uniform(1e7, 1e8, n)
    base = par_digests(pool, h)
    smallest = np.full(n, np.inf)
    for d in ladder:
        todo = np.where(np.isinf(smallest))[0]
        if todo.size == 0:
            break
        k = par_digests(pool, h[todo] + d)
        for j, idx in enumerate(todo):
            if k[j] != base[idx]:
                smallest[idx] = d
    k8 = par_digests(pool, h + 1e-8)
    fin = smallest[np.isfinite(smallest)]
    return {"n": n, "median_smallest_perturbation": float(np.median(fin)),
            "never_changed": int(np.isinf(smallest).sum()),
            "unchanged_by_1e-8": float(np.mean([a == b for a, b in zip(base, k8)]))}


def exp_throughput(pool, workers, n_single=60_000, n_par=240_000):
    """Single-process vs pool throughput on identical work, minimal IPC."""
    target = b"\x00" * 32
    base = T0_EXAMPLE + np.arange(n_par) * 1e-6
    pool.map(w_count_matches, [(c, target) for c in chunks(base[:4000], 1000)])  # warm-up
    s = time.perf_counter()
    for c in chunks(base[:n_single], 20_000):
        w_count_matches((c, target))
    single = n_single / (time.perf_counter() - s)
    s = time.perf_counter()
    pool.map(w_count_matches, [(c, target) for c in chunks(base, 20_000)])
    par = n_par / (time.perf_counter() - s)
    return {"workers": workers, "single_keys_per_s": single, "pool_keys_per_s": par,
            "speedup": par / single}


def exp_raw_stats(pool, rng, n):
    h = rng.uniform(1e7, 1e8, n)
    counts = sum(pool.map(w_bincount, chunks(h, 2_500))).astype(np.float64)
    p = counts / counts.sum()
    nz = p[p > 0]
    exp = counts.sum() / 256
    pos = counts[counts > 0]
    return {"n": n, "entropy_bits_per_byte": float(-(nz * np.log2(nz)).sum()),
            "never_observed": int((counts == 0).sum()),
            "max_over_min": float(pos.max() / pos.min()),
            "chi2": float(((counts - exp) ** 2 / exp).sum())}


def exp_precision(pool, rng, n):
    fi = np.finfo(np.longdouble)
    h = rng.uniform(1e7, 1e8, n)
    res = pool.map(w_precision, chunks(h, 2_500))
    mism = np.concatenate([r[0] for r in res])
    diffs = np.concatenate([r[1] for r in res])
    return {"n": n, "longdouble_mantissa_bits": int(fi.nmant) + 1,
            "longdouble_is_x87_80bit": bool(fi.nmant == 63),
            "mismatching": int(mism.sum()), "rate": float(mism.mean()),
            "mean_differing_bytes": float(diffs[mism].mean()) if mism.any() else 0.0}


def exp_attack_coverage(pool, rng, window=0.02, n_secrets=20_000):
    """
    How good is a naive grid search?  Reference = every float64 in a window
    (step = 1 ulp), which finds essentially every distinct key. Coarser grids
    are strided subsamples. For each stride we report
      * distinct-key coverage (fraction of cells visited), and
      * hit probability: fraction of uniformly random secrets in the window
        whose key is found, i.e. the attacker's success probability.
    """
    npts = int(window / ULP)
    base = float(np.floor(T0_EXAMPLE / ULP) * ULP)
    ref_t = base + np.arange(npts, dtype=np.float64) * ULP     # exact, 1 ulp apart
    ref = par_digests(pool, ref_t, size=16)
    ref_set = set(ref)
    secrets = rng.uniform(base, base + npts * ULP, n_secrets)
    skeys = par_digests(pool, secrets, size=16)
    out = {"window": window, "ref_points": npts, "ref_distinct": len(ref_set),
           "ref_density_per_unit": len(ref_set) / (npts * ULP),
           "secrets_found_by_reference": float(np.mean([k in ref_set for k in skeys])),
           "strides": []}
    for stride in (7, 22, 67, 201, 671):                      # ~1e-7 ... 1e-5
        sub = set(ref[::stride])
        step = stride * ULP
        out["strides"].append({
            "stride_ulps": stride, "step": step,
            "cell_coverage": len(sub) / len(ref_set),
            "hit_probability": float(np.mean([k in sub for k in skeys])),
            "evals_per_unit_t": 1.0 / step,
        })
    return out


# --------------------------------------------------------------------------
def tex_num(x, digits=3):
    """Format x as LaTeX scientific notation without the dollar signs."""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, digits - 1) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.{digits - 1}f}\\times10^{{{e}}}"


def write_macros(R, path):
    t, g1, g2 = R["throughput"], R["grid_1e-6"], R["grid_1e-7"]
    rs, pr, pe, ex = R["raw"], R["precision"], R["perturbation"], R["extrapolation"]
    ac = R["attack_coverage"]
    best = R["attack"]["chosen"]
    M = {
        "GridA": f"{g1['distinct']:,}".replace(",", "{,}"),
        "GridB": f"{g2['distinct']:,}".replace(",", "{,}"),
        "Density": tex_num(R["density_mean"], 2),
        "MedianPerturb": "10^{%d}" % round(math.log10(pe["median_smallest_perturbation"])),
        "Unchanged": f"{100 * pe['unchanged_by_1e-8']:.0f}",
        "KeysTotal": tex_num(ex["keys_mean"], 2),
        "LogKeys": f"{math.log2(ex['keys_mean']):.0f}",
        "ThrOne": tex_num(t["single_keys_per_s"], 2),
        "ThrPool": tex_num(t["pool_keys_per_s"], 2),
        "Speedup": f"{t['speedup']:.2f}",
        "Workers": str(t["workers"]),
        "YearsOneLow": f"{ex['years_ideal_one_core_low']:.1f}",
        "YearsOneHigh": f"{ex['years_ideal_one_core_high']:.1f}",
        "YearsPoolLow": f"{ex['years_ideal_pool_low']:.1f}",
        "YearsPoolHigh": f"{ex['years_ideal_pool_high']:.1f}",
        "Entropy": f"{rs['entropy_bits_per_byte']:.3f}",
        "Never": str(rs["never_observed"]),
        "Ratio": f"{rs['max_over_min']:.2f}",
        "Chi": tex_num(rs["chi2"], 2),
        "MismatchPct": f"{100 * pr['rate']:.2f}",
        "MismatchN": str(pr["mismatching"]),
        "MismatchOneIn": str(int(round(1 / pr["rate"]))) if pr["rate"] else "0",
        "DiffBytes": f"{pr['mean_differing_bytes']:.1f}",
        "GridStep": "10^{-6}",
        "GridHit": f"{100 * best['hit_probability']:.1f}",
        "GridCells": f"{100 * best['cell_coverage']:.0f}",
        "GridEvals": tex_num(ex["grid_evals_total"], 2),
        "GridYearsPool": f"{ex['grid_years_pool']:.0f}",
        "CoverageRows": "".join(
            "$%s$ & %.1f\\%% & %.1f\\%% \\\\\n" % (tex_num(c["step"], 2), 100 * c["cell_coverage"],
                                                  100 * c["hit_probability"])
            for c in ac["strides"]),
        "Platform": ex["cpu_model"].replace("_", "\\_").replace("&", "\\&"),
    }
    with open(path, "w") as f:
        f.write("% Auto-generated by zkd_colab.py -- do not edit by hand.\n")
        for k, v in M.items():
            f.write(f"\\renewcommand{{\\zkd{k}}}{{{v}}}\n")


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default="results_colab.json")
    ap.add_argument("--macros", default="results_macros.tex")
    a = ap.parse_args()
    sc = 0.05 if a.quick else 1.0
    S = lambda n: max(200, int(n * sc))

    rng = np.random.default_rng(a.seed)
    t_start = time.time()
    R = {"meta": {"seed": a.seed, "quick": a.quick, "workers": a.workers,
                  "python": sys.version.split()[0], "numpy": np.__version__,
                  "platform": platform.platform(), "cpu_model": cpu_model(),
                  "logical_cpus": os.cpu_count()}}
    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    with ctx.Pool(a.workers) as pool:
        def step(msg):
            print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)

        step("density formulas");              R["density"] = exp_density_check()
        step("grid 200k @1e-6");               R["grid_1e-6"] = exp_grid(pool, 1e-6, S(200_000))
        step("grid 400k @1e-7");               R["grid_1e-7"] = exp_grid(pool, 1e-7, S(400_000))
        step("perturbation ladder");           R["perturbation"] = exp_perturbation(pool, rng, S(300))
        step("raw byte statistics");           R["raw"] = exp_raw_stats(pool, rng, S(20_000))
        step("float64 vs longdouble");         R["precision"] = exp_precision(pool, rng, S(20_000))
        step("throughput, 1 vs %d workers" % a.workers)
        R["throughput"] = exp_throughput(pool, a.workers, S(60_000), S(240_000))
        step("attack coverage (grid vs ideal)")
        R["attack_coverage"] = exp_attack_coverage(pool, rng, 0.02 if not a.quick else 0.004,
                                                   S(20_000))

    d1 = R["grid_1e-6"]["distinct"] / R["grid_1e-6"]["width"]
    d2 = R["grid_1e-7"]["distinct"] / R["grid_1e-7"]["width"]
    R["density_mean"] = (d1 + d2) / 2
    width = 9e7
    keys = {"low": min(d1, d2) * width, "high": max(d1, d2) * width,
            "mean": R["density_mean"] * width}
    one, pool_rate = R["throughput"]["single_keys_per_s"], R["throughput"]["pool_keys_per_s"]
    chosen = next(s for s in R["attack_coverage"]["strides"] if s["stride_ulps"] == 67)  # ~1e-6
    grid_evals = width * chosen["evals_per_unit_t"]
    R["extrapolation"] = {
        "range": [1e7, 1e8], "keys_low": keys["low"], "keys_high": keys["high"],
        "keys_mean": keys["mean"], "log2_keys_mean": math.log2(keys["mean"]),
        "years_ideal_one_core_low": keys["low"] / one / SECONDS_PER_YEAR,
        "years_ideal_one_core_high": keys["high"] / one / SECONDS_PER_YEAR,
        "years_ideal_pool_low": keys["low"] / pool_rate / SECONDS_PER_YEAR,
        "years_ideal_pool_high": keys["high"] / pool_rate / SECONDS_PER_YEAR,
        "grid_step": chosen["step"], "grid_evals_total": grid_evals,
        "grid_hit_probability": chosen["hit_probability"],
        "grid_years_pool": grid_evals / pool_rate / SECONDS_PER_YEAR,
        "cpu_model": R["meta"]["cpu_model"],
        "caveat": "Density measured near t0=%.7f and extrapolated." % T0_EXAMPLE,
    }
    R["attack"] = {"chosen": chosen}
    R["meta"]["runtime_s"] = time.time() - t_start
    write_macros(R, a.macros)
    with open(a.json, "w") as f:
        json.dump(R, f, indent=2)

    ex, pr, rs, th, ac = R["extrapolation"], R["precision"], R["raw"], R["throughput"], R["attack_coverage"]
    print("\n=== Summary ===")
    print(f"CPU: {R['meta']['cpu_model']}  ({R['meta']['logical_cpus']} logical CPUs, {a.workers} workers)")
    print(f"distinct keys: {R['grid_1e-6']['distinct']} / {R['grid_1e-7']['distinct']}; density {R['density_mean']:.3g}/unit")
    print(f"keys in [1e7,1e8]: {ex['keys_mean']:.3g} = 2^{ex['log2_keys_mean']:.1f}")
    print(f"throughput: {th['single_keys_per_s']:.3g}/s single, {th['pool_keys_per_s']:.3g}/s with {a.workers} workers (x{th['speedup']:.2f})")
    print(f"ideal enumeration: {ex['years_ideal_one_core_low']:.1f}-{ex['years_ideal_one_core_high']:.1f} y (1 core), "
          f"{ex['years_ideal_pool_low']:.1f}-{ex['years_ideal_pool_high']:.1f} y ({a.workers} workers)")
    print("grid attack coverage (stride ulps, step, cells found, secret hit prob):")
    for s in ac["strides"]:
        print(f"   {s['stride_ulps']:4d}  {s['step']:.1e}  {100*s['cell_coverage']:5.1f}%  {100*s['hit_probability']:5.1f}%")
    print(f"grid @1e-6: {ex['grid_evals_total']:.2g} evals, hit prob {100*ex['grid_hit_probability']:.0f}%, "
          f"{ex['grid_years_pool']:.0f} y on {a.workers} workers")
    print(f"raw: H={rs['entropy_bits_per_byte']:.3f} never={rs['never_observed']} ratio={rs['max_over_min']:.2f} chi2={rs['chi2']:.3g}")
    print(f"precision: {pr['mismatching']}/{pr['n']} = {100*pr['rate']:.2f}% (x87={pr['longdouble_is_x87_80bit']})")
    print(f"\nWrote {a.json} and {a.macros}  ({R['meta']['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()

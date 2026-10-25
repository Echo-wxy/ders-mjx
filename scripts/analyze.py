"""Summaries of all experiments (tables as CSV and Markdown) from ``results/``.

Statistics
----------
* Gradient metrics are first averaged within each seed and then reported as
  mean and standard deviation across seeds.
* Paired comparisons (G6) use identical problems for every method. The
  difference to DERS is computed per problem; a cluster bootstrap resamples
  seeds (and problems within each resampled seed) 10,000 times to give a 95%
  interval of the mean difference. The fraction of problems on which DERS is
  better is reported alongside.

Usage::

    python scripts/analyze.py --results results --out analysis
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
NEAR_PHI = 0.005   # m: near-contact threshold
WINDOW = 3         # steps: active-set change within +-WINDOW counts as near contact
BAND = 0.01        # m: pre-contact band (contact candidates exist)
OBJ_ROWS = [3, 4, 5, 8, 9, 10, 11]  # object position and velocity rows of the 12-dim state


def cos(a, b):
    a, b = np.ravel(a), np.ravel(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return np.nan if na == 0 or nb == 0 else float(a @ b / (na * nb))


# ----------------------------------------------------------------- G2 / G3 / G4
def choose_h(z, far):
    """Finite-difference step: highest median far-from-contact self-consistency
    between neighbouring step sizes; ties go to the larger step."""
    hs = z["hs"]
    scores = []
    for k in range(len(hs) - 1):
        scores.append(np.nanmedian([cos(z["fd"][i, k], z["fd"][i, k + 1]) for i in np.where(far)[0]]))
    scores.append(-np.inf)
    best = max(scores)
    k = min(i for i, s in enumerate(scores) if s >= best - 1e-6)
    return k, scores


def strata(z):
    seed, t, phi, act = z["seed"], z["t"], z["phi"], z["active"]
    change = np.zeros(len(t), bool)
    for s in np.unique(seed):
        idx = np.where(seed == s)[0]
        order = idx[np.argsort(t[idx])]
        ch = np.r_[False, (act[order][1:] != act[order][:-1]).any(1)]
        tt = t[order]
        for j in np.where(ch)[0]:
            change[order[np.abs(tt - tt[j]) <= WINDOW]] = True
    near = (phi <= NEAR_PHI) | change
    gap = z["gap"] if "gap" in z else np.full(len(t), np.nan)
    pre = (gap > 0.0) & (gap <= BAND)  # separated, inside the contact-candidate band
    return near, ~near, pre


def g3_tables(path):
    with np.load(path, allow_pickle=True) as f:
        z = {k: f[k] for k in f.files}  # decompress once (NpzFile re-reads on every access)
    scene = os.path.basename(path).split("_", 1)[1].rsplit(".", 1)[0]
    near, far, pre = strata(z)
    k, sc = choose_h(z, far)
    switch = z["switch"][:, k]
    fd = z["fd"][:, k]
    groups = {"all": np.ones(len(far), bool), "far": far, "near": near, "switch": switch, "pre_contact": pre}
    methods = [str(m) for m in z["methods"]] + [str(v) for v in z["variants"]]
    J = {m: z[f"J_{m}"] for m in methods}
    for si, sg in enumerate(z["sigmas"]):
        J[f"rs_sigma{sg:g}"] = z["rs"][:, si]
    rows = []
    for m, Jm in J.items():
        for g, mask in groups.items():
            per_seed = []
            for s in np.unique(z["seed"]):
                idx = np.where(mask & (z["seed"] == s))[0]
                if idx.size == 0:
                    continue
                c = np.array([cos(Jm[i], fd[i]) for i in idx])
                ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
                rmse = np.array([np.sqrt(np.mean((Jm[i] - fd[i]) ** 2)) for i in idx])
                ratio = np.array([np.linalg.norm(Jm[i]) / max(np.linalg.norm(fd[i]), 1e-300) for i in idx])
                inform = np.array([np.linalg.norm(Jm[i][OBJ_ROWS]) > 1e-8 for i in idx])
                per_seed.append(dict(cos=np.nanmean(c), angle=np.nanmean(ang), angle_p99=np.nanpercentile(ang, 99),
                                     rmse=rmse.mean(), sign_flip=np.mean(c < 0) * 100, norm_ratio=np.median(ratio),
                                     object_grad_nonzero=inform.mean() * 100, n=idx.size))
            if not per_seed:
                rows.append(dict(scene=scene, method=m, stratum=g, n=0))
                continue
            df = pd.DataFrame(per_seed)
            row = dict(scene=scene, method=m, stratum=g, n=int(df.n.sum()), h_star=float(z["hs"][k]))
            for col in ["cos", "angle", "angle_p99", "rmse", "sign_flip", "norm_ratio", "object_grad_nonzero"]:
                row[col] = df[col].mean()
                row[col + "_sd"] = df[col].std(ddof=1) if len(df) > 1 else np.nan
            rows.append(row)
    fdrows = []  # G2: step-size table
    for kk, h in enumerate(z["hs"]):
        for g, mask in (("far", far), ("near", near)):
            idx = np.where(mask)[0]
            fdrows.append(dict(scene=scene, h=float(h), stratum=g,
                               self_consistency=sc[kk] if g == "far" else np.nanmedian(
                                   [cos(z["fd"][i, kk], z["fd"][i, min(kk + 1, len(z["hs"]) - 1)]) for i in idx]),
                               switch_pct=float(z["switch"][:, kk].mean() * 100),
                               ders_cos=np.nanmean([cos(z["J_ders"][i], z["fd"][i, kk]) for i in idx]) if "J_ders" in z else np.nan,
                               hard_cos=np.nanmean([cos(z["J_hard_ad"][i], z["fd"][i, kk]) for i in idx]) if "J_hard_ad" in z else np.nan))
    extra = {}
    for v in z["variants"]:
        extra[str(v)] = float(np.mean(z[f"fwd_bitwise_{v}"]) * 100)
    return pd.DataFrame(rows), pd.DataFrame(fdrows), extra


# ----------------------------------------------------------------------- G6
def cluster_bootstrap(df, value, ref="ders", n_boot=10_000):
    """Mean paired difference (method - ref) with a seed-clustered bootstrap 95% CI."""
    out = []
    key = ["scene", "variant", "seed", "problem"]
    piv = df.pivot_table(index=key, columns="method", values=value)
    if ref not in piv:
        return pd.DataFrame()
    for m in piv.columns:
        if m == ref:
            continue
        d = (piv[m] - piv[ref]).dropna()
        for (scene, variant), dd in d.groupby(level=[0, 1]):
            seeds = dd.index.get_level_values(2).unique()
            by_seed = [dd.xs(s, level=2).values for s in seeds]
            P = min(len(x) for x in by_seed)
            arr = np.stack([x[:P] for x in by_seed])          # seeds x problems
            S = arr.shape[0]
            si = RNG.integers(0, S, (n_boot, S))
            pi = RNG.integers(0, P, (n_boot, S, P))
            boots = arr[si[:, :, None], pi].mean(axis=(1, 2))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out.append(dict(scene=scene, variant=variant, method=m, reference=ref, measure=value,
                            mean_diff=dd.mean(), ci_low=lo, ci_high=hi, n_pairs=len(dd), n_seeds=len(seeds),
                            ref_better_pct=float((dd > 0).mean() * 100)))
    return pd.DataFrame(out)


def g6_tables(res):
    tabs = {}
    a = [pd.read_csv(f) for f in glob.glob(os.path.join(res, "g6a", "*.csv"))]
    if a:
        a = pd.concat(a)
        eta = "dcost_eta0.3" if "dcost_eta0.3" in a.columns else "dcost_eta0.1"
        a["descent"] = (a[eta] < 0).astype(float)
        summ = a.groupby(["scene", "variant", "method"]).agg(
            dcost=(eta, "mean"), descent_pct=("descent", lambda x: 100 * x.mean()),
            grad_zero_pct=("grad_norm", lambda x: 100 * (x == 0).mean()), n=("problem", "size")).reset_index()
        fdcols = [c for c in a.columns if c.startswith("cos_fd")]
        if fdcols:
            summ = summ.merge(a.groupby(["scene", "variant", "method"])[fdcols].mean().reset_index(), how="left")
        tabs["g6a_summary"] = summ
        tabs["g6a_paired"] = cluster_bootstrap(a, eta)
    b = [pd.read_csv(f) for f in glob.glob(os.path.join(res, "g6b", "*.csv"))]
    if b:
        b = pd.concat(b)
        # equal simulation-call budget: cost of each method after the number of
        # iterations that uses as many simulation calls as the AD methods
        ad_calls = 3 * (b["curve"].str.count(";").max())
        b["cost_equal_calls"] = [float(c.split(";")[min(int(ad_calls // k), c.count(";"))])
                                 for c, k in zip(b["curve"], b["calls_per_iter"])]
        tabs["g6b_summary"] = b.groupby(["scene", "variant", "method"]).agg(
            success_pct=("success", lambda x: 100 * x.mean()), final_dist=("final_dist", "median"),
            cost_final=("cost_final", "mean"), cost_equal_calls=("cost_equal_calls", "mean"),
            n=("problem", "size")).reset_index()
        tabs["g6b_paired_cost"] = cluster_bootstrap(b, "cost_final")
        tabs["g6b_paired_success"] = cluster_bootstrap(b, "success")
        tabs["g6b_paired_equal_calls"] = cluster_bootstrap(b, "cost_equal_calls")
    g5 = [pd.read_csv(f) for f in glob.glob(os.path.join(res, "g5", "*.csv"))]
    if g5:
        g5 = pd.concat(g5)
        cols = [c for c in g5.columns if c.startswith("cos_fd")]
        tabs["g5_multistep"] = g5.groupby(["scene", "horizon", "method"])[cols].agg(["mean", "std"]).reset_index()
    return tabs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="analysis")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tabs = {}
    g3, g2, g4 = [], [], {}
    for f in sorted(glob.glob(os.path.join(args.results, "g3", "*.npz"))):
        t3, t2, _ = g3_tables(f)
        g3.append(t3)
        g2.append(t2)
    if g3:
        tabs["g3_gradient_strata"] = pd.concat(g3)
        tabs["g2_fd_step"] = pd.concat(g2)
    for f in sorted(glob.glob(os.path.join(args.results, "g4", "*.npz"))):
        t4, _, bit = g3_tables(f)
        t4["fwd_bitwise_pct"] = t4["method"].map(bit)
        g4[f] = t4
    if g4:
        tabs["g4_temperature"] = pd.concat(g4.values())
    g1 = [pd.read_csv(f) for f in glob.glob(os.path.join(args.results, "g1", "*.csv"))]
    if g1:
        g1 = pd.concat(g1)
        tabs["g1_forward"] = g1.groupby(["scene", "method", "coupling"]).agg(
            step_bitwise_pct=("step_bitwise_frac", lambda x: 100 * x.mean()),
            rollout_bitwise_pct=("rollout_bitwise_frac", lambda x: 100 * x.mean()),
            step_mean_rel_err=("step_mean_rel_err", "mean"),
            step_mismatch_mean_rel_err=("step_mismatch_mean_rel_err", "mean"),
            rollout_final_rel_err=("rollout_final_rel_err", "mean"),
            obj_final_drift_m=("obj_final_drift_m", "mean"), n_states=("state", "size")).reset_index()
    tabs.update(g6_tables(args.results))
    g7 = [pd.read_csv(f) for f in glob.glob(os.path.join(args.results, "g7", "*.csv"))]
    if g7:
        tabs["g7_cost"] = pd.concat(g7)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as fh:
        for name, df in tabs.items():
            df.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
            fh.write(f"## {name}\n\n{df.to_markdown(index=False, floatfmt='.4g')}\n\n")
    print(f"wrote {len(tabs)} tables to {args.out}")


if __name__ == "__main__":
    main()

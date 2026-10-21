#!/usr/bin/env python3
"""Brightness-bin coefficient / effect-size stability analysis."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.chdir("./runs")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

OUT = "bright_fine"
O = "./open"
y = pd.read_csv(f"{O}/train.csv").hardness.values
n = len(y)
assert n == 500

# ---- load features ----
def load_feats(name):
    path = f"{OUT}/bf_feats.csv" if name == "orig" else f"{OUT}/bf_feats_{name}.csv"
    return pd.read_csv(path).iloc[:500].drop(columns="ID")

XF = {m: load_feats(m) for m in ["orig", "clahe"]}

# fine bin sets (area ratios only)
RZ24 = [f"R_z{s}{abs(v):.2f}" for v in np.arange(-3.0, 3.0, 0.25)
        for s in (["-"] if v < 0 else ["+"])]
# fix naming to match CSV
RZ24 = [c for c in XF["orig"].columns if c.startswith("R_z") and "q" not in c]
A16 = [c for c in XF["orig"].columns if c.startswith("A_h16_")]
A32 = [c for c in XF["orig"].columns if c.startswith("A_h32_")]
assert len(RZ24) == 24 and len(A16) == 16 and len(A32) == 32, (len(RZ24), len(A16), len(A32))


def merge_adjacent(X, cols, group_size, prefix):
    """Sum adjacent area-ratio columns into coarser bins."""
    out = {}
    names = []
    for i in range(0, len(cols), group_size):
        chunk = cols[i:i + group_size]
        if not chunk:
            continue
        # label from first..last token
        a = chunk[0].replace(prefix, "")
        b = chunk[-1].replace(prefix, "")
        nm = f"{prefix}{a}~{b}" if a != b else f"{prefix}{a}"
        out[nm] = X[chunk].sum(axis=1).values
        names.append(nm)
    return pd.DataFrame(out, index=X.index), names


# coarse: relative 24→8 (×3), 24→6 (×4); absolute 16→8 (×2)
def make_sets(X):
    sets = {
        "상대_fine24": RZ24,
        "절대16_fine": A16,
        "절대32_fine": A32,
    }
    # build coarse as synthetic frames attached via a side dict
    coarse = {}
    df8, n8 = merge_adjacent(X, RZ24, 3, "R_z")
    df6, n6 = merge_adjacent(X, RZ24, 4, "R_z")
    da8, na8 = merge_adjacent(X, A16, 2, "A_h16_")
    coarse["상대_coarse8"] = df8
    coarse["상대_coarse6"] = df6
    coarse["절대16_coarse8"] = da8
    return sets, coarse


ALPHAS = np.logspace(-1, 6, 30)


def fit_ridge_coefs(Xmat, yv):
    """Standardize, RidgeCV, return coef in original feature order (on standardized X)."""
    sc = StandardScaler()
    Z = sc.fit_transform(Xmat)
    # drop zero-var (already handled by scaler with care)
    md = RidgeCV(ALPHAS).fit(Z, yv)
    return md.coef_.copy(), float(md.alpha_)


def get_X(Xdf, cols=None, coarse_df=None):
    if coarse_df is not None:
        M = coarse_df.values.astype(float)
        names = list(coarse_df.columns)
    else:
        M = Xdf[cols].values.astype(float)
        names = list(cols)
    # drop zero std cols but keep alignment via mask
    std = M.std(0)
    keep = std > 1e-12
    return M[:, keep], [nm for nm, k in zip(names, keep) if k]


# ============================================================
# 1) Repeated CV coefficient stability
# ============================================================
print("=== 1) Repeated CV coef stability ===", flush=True)
cv_rows = []  # one row per (tf, set, bin, fold_idx) — actually store arrays
cv_store = {}  # key -> list of coef Series

configs = []
for tf in ["orig", "clahe"]:
    Xdf = XF[tf]
    fine, coarse = make_sets(Xdf)
    for sn, cols in fine.items():
        configs.append((tf, sn, Xdf, cols, None))
    for sn, cdf in coarse.items():
        configs.append((tf, sn, Xdf, None, cdf))

for tf, sn, Xdf, cols, cdf in configs:
    key = f"{tf}|{sn}"
    coef_list = []
    alphas = []
    name_ref = None
    for s in range(10):
        kf = KFold(5, shuffle=True, random_state=100 + s)
        for tr, _va in kf.split(np.zeros(n)):
            M, names = get_X(Xdf, cols, cdf)
            if name_ref is None:
                name_ref = names
            # re-get with same columns if some fold has zero-var — use global keep
            M_all, names_all = get_X(Xdf, cols, cdf)
            # use all-data keep mask for consistent columns
            coef, alpha = fit_ridge_coefs(M_all[tr], y[tr])
            coef_list.append(pd.Series(coef, index=names_all))
            alphas.append(alpha)
    C = pd.DataFrame(coef_list)
    mean = C.mean(0)
    std = C.std(0)
    # sign consistency: fraction with same sign as mean (zeros count as inconsistent if mean!=0)
    sign_ok = ((np.sign(C) == np.sign(mean.replace(0, np.nan))).sum(0) / len(C))
    # also fraction positive
    frac_pos = (C > 0).sum(0) / len(C)
    for b in C.columns:
        cv_rows.append({
            "tf": tf, "set": sn, "bin": b,
            "mean": mean[b], "std": std[b],
            "sign_consist": sign_ok[b], "frac_pos": frac_pos[b],
            "n_fits": len(C), "alpha_med": float(np.median(alphas)),
        })
    cv_store[key] = C
    print(f"  {key}: {len(C)} fits, {C.shape[1]} bins, alpha_med={np.median(alphas):.1f}", flush=True)

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(f"{OUT}/stability_cv_coefs.csv", index=False)

# ============================================================
# 2) Bootstrap CI for coefs + effect sizes
# ============================================================
print("=== 2) Bootstrap 1000 ===", flush=True)
rng = np.random.default_rng(42)
B = 1000
boot_coef_rows = []
boot_d_rows = []

# which sets to bootstrap for coefs: fine relative & abs16 for orig/clahe
boot_coef_cfgs = [
    ("orig", "상대_fine24", RZ24, None),
    ("orig", "절대16_fine", A16, None),
    ("clahe", "상대_fine24", RZ24, None),
    ("clahe", "절대16_fine", A16, None),
]
# also coarse8 relative
for tf in ["orig", "clahe"]:
    _, coarse = make_sets(XF[tf])
    boot_coef_cfgs.append((tf, "상대_coarse8", None, coarse["상대_coarse8"]))

boot_coef_arr = {}  # key -> (B, p) array + names
for tf, sn, cols, cdf in boot_coef_cfgs:
    Xdf = XF[tf]
    M_all, names = get_X(Xdf, cols, cdf)
    arr = np.zeros((B, len(names)))
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        arr[b], _ = fit_ridge_coefs(M_all[idx], y[idx])
    boot_coef_arr[f"{tf}|{sn}"] = (arr, names)
    lo = np.percentile(arr, 2.5, axis=0)
    hi = np.percentile(arr, 97.5, axis=0)
    mean = arr.mean(0)
    for i, nm in enumerate(names):
        boot_coef_rows.append({
            "tf": tf, "set": sn, "bin": nm,
            "mean": mean[i], "ci_lo": lo[i], "ci_hi": hi[i],
            "sign_stable": (lo[i] > 0) or (hi[i] < 0),
        })
    print(f"  boot coef {tf}|{sn}", flush=True)

pd.DataFrame(boot_coef_rows).to_csv(f"{OUT}/stability_boot_coefs.csv", index=False)

# effect size d bootstrap for ALL features (orig + clahe), top features
# Cohen's d hi vs ctl; ctl = 170..230 inclusive, hi threshold varies later
def cohen_d(Xa, Xb):
    # Xa, Xb: (n_a, p), (n_b, p)
    ma, mb = Xa.mean(0), Xb.mean(0)
    va, vb = Xa.var(0, ddof=1), Xb.var(0, ddof=1)
    sp = np.sqrt((va + vb) / 2)
    sp = np.where(sp < 1e-12, np.nan, sp)
    return (ma - mb) / sp


def feat_matrix(tf):
    return XF[tf].values.astype(float), list(XF[tf].columns)


# bootstrap d at threshold 230 for orig (all feats) — store for top10 + all
print("  boot d ...", flush=True)
for tf in ["orig", "clahe"]:
    M, names = feat_matrix(tf)
    # point estimate
    hi_m = y > 230
    ctl_m = (y >= 170) & (y <= 230)
    d0 = cohen_d(M[hi_m], M[ctl_m])
    # bootstrap: resample within groups to keep group sizes, AND full resample then redefine groups
    # Use stratified (resample hi and ctl separately) for CI of d given the definition
    hi_idx = np.where(hi_m)[0]
    ctl_idx = np.where(ctl_m)[0]
    d_boot = np.zeros((B, len(names)))
    for b in range(B):
        hi_s = rng.choice(hi_idx, size=len(hi_idx), replace=True)
        ctl_s = rng.choice(ctl_idx, size=len(ctl_idx), replace=True)
        d_boot[b] = cohen_d(M[hi_s], M[ctl_s])
    lo = np.nanpercentile(d_boot, 2.5, axis=0)
    hi = np.nanpercentile(d_boot, 97.5, axis=0)
    # top 10 by |d0|
    order = np.argsort(-np.abs(d0))
    for rank, i in enumerate(order[:10], 1):
        boot_d_rows.append({
            "tf": tf, "thr": 230, "rank": rank, "feat": names[i],
            "type": "상대" if names[i].startswith("R_") else "절대",
            "d": d0[i], "ci_lo": lo[i], "ci_hi": hi[i],
            "ci_excludes_0": (lo[i] > 0) or (hi[i] < 0),
            "n_hi": int(hi_m.sum()), "n_ctl": int(ctl_m.sum()),
        })
    # also save full for later
    np.save(f"{OUT}/boot_d_{tf}_ thr230.npy".replace(" ", ""), d_boot)
    print(f"  boot d {tf} done", flush=True)

pd.DataFrame(boot_d_rows).to_csv(f"{OUT}/stability_boot_d.csv", index=False)

# ============================================================
# 3) Threshold sensitivity
# ============================================================
print("=== 3) Threshold sensitivity ===", flush=True)
thr_rows = []
rank_rows = []
for thr in [230, 220, 210]:
    hi_m = y > thr
    ctl_m = (y >= 170) & (y <= thr)  # control below threshold, above 170
    # also keep original ctl 170-230 for reference? Task: 고경도 기준 바꾸기 — control as non-hi in mid range
    # Use ctl = (y >= 170) & (y <= 230) & ~hi for fair mid-band, or simply not hi & in [170,230]
    # Cleaner: ctl = (hardness >= 170) & (hardness <= 230) always the mid band,
    # and hi = hardness > thr. When thr=210, hi overlaps ctl — bad.
    # Better: ctl = (170 <= y <= min(230, thr)) wait.
    # Standard: hi = y > thr, ctl = (y >= 170) & (y <= 230) & (y <= thr) 
    # For thr=230: ctl 170-230, but y<=230 & y>230 empty overlap — ctl should exclude hi.
    # ctl = (y >= 170) & (y <= 230) & ~hi_m
    ctl_m = (y >= 170) & (y <= 230) & (~hi_m)
    # When thr=210, hi includes 210-230 which were in old ctl; ctl shrinks to 170-210.
    # Actually if thr=210, hi=y>210 includes some in 210-230. ctl = 170-230 & y<=210 = 170-210.
    n_hi, n_ctl = int(hi_m.sum()), int(ctl_m.sum())
    thr_rows.append({"thr": thr, "n_hi": n_hi, "n_ctl": n_ctl})
    for tf in ["orig", "clahe"]:
        M, names = feat_matrix(tf)
        d = cohen_d(M[hi_m], M[ctl_m])
        order = np.argsort(-np.abs(d))
        for rank, i in enumerate(order[:15], 1):
            rank_rows.append({
                "thr": thr, "tf": tf, "rank": rank, "feat": names[i],
                "type": "상대" if names[i].startswith("R_") else "절대",
                "d": d[i], "n_hi": n_hi, "n_ctl": n_ctl,
            })
        print(f"  thr>{thr} {tf}: n_hi={n_hi} n_ctl={n_ctl} top={names[order[0]]} d={d[order[0]]:.3f}", flush=True)

pd.DataFrame(thr_rows).to_csv(f"{OUT}/stability_thr_n.csv", index=False)
rank_df = pd.DataFrame(rank_rows)
rank_df.to_csv(f"{OUT}/stability_thr_ranks.csv", index=False)

# overlap of top-10 features across thresholds
overlap_rows = []
for tf in ["orig", "clahe"]:
    sets = {}
    for thr in [230, 220, 210]:
        sub = rank_df[(rank_df.tf == tf) & (rank_df.thr == thr) & (rank_df.rank <= 10)]
        sets[thr] = set(sub.feat)
    for a, b in [(230, 220), (230, 210), (220, 210)]:
        inter = sets[a] & sets[b]
        overlap_rows.append({
            "tf": tf, "pair": f"{a}∩{b}", "n_overlap": len(inter),
            "feats": ", ".join(sorted(inter)),
        })
pd.DataFrame(overlap_rows).to_csv(f"{OUT}/stability_thr_overlap.csv", index=False)

# ============================================================
# 4) Figures: coefficient error bars
# ============================================================
print("=== 4) Figures ===", flush=True)

def plot_coef_err(key_cv, key_boot, title, out_path, max_bins=32):
    """Error bars = CV std; overlay bootstrap 95% CI as thinner caps if available."""
    sub = cv_df[(cv_df.tf + "|" + cv_df.set) == key_cv].copy()
    if sub.empty:
        # try match
        parts = key_cv.split("|")
        sub = cv_df[(cv_df.tf == parts[0]) & (cv_df.set == parts[1])].copy()
    if sub.empty:
        print("  empty", key_cv)
        return
    # sort by bin name order as in data
    C = cv_store[key_cv]
    order = list(C.columns)
    sub = sub.set_index("bin").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(sub)), 4.5))
    x = np.arange(len(sub))
    colors = ["tab:red" if v > 0 else "tab:blue" for v in sub["mean"]]
    ax.bar(x, sub["mean"], yerr=sub["std"], color=colors, alpha=0.75,
           error_kw=dict(ecolor="k", lw=0.8, capsize=2))
    # bootstrap CI ticks
    bsub = pd.DataFrame(boot_coef_rows)
    bsub = bsub[(bsub.tf + "|" + bsub.set) == key_boot] if key_boot else pd.DataFrame()
    if not bsub.empty:
        bsub = bsub.set_index("bin")
        for i, b in enumerate(order):
            if b in bsub.index:
                lo, hi = bsub.loc[b, "ci_lo"], bsub.loc[b, "ci_hi"]
                ax.plot([i, i], [lo, hi], color="orange", lw=1.2, zorder=5)
                ax.plot([i - 0.15, i + 0.15], [lo, lo], color="orange", lw=1.0)
                ax.plot([i - 0.15, i + 0.15], [hi, hi], color="orange", lw=1.0)
    ax.axhline(0, color="k", lw=0.6)
    labels = [str(b).replace("R_z", "").replace("A_h16_", "").replace("A_h32_", "") for b in order]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("표준화 Ridge 계수")
    ax.set_title(title + "\n검정=CV std(50 fits), 주황=bootstrap 95% CI")
    # annotate sign consistency for unstable
    for i, r in sub.iterrows():
        if r["sign_consist"] < 0.8:
            ax.text(i, r["mean"] + (1 if r["mean"] >= 0 else -1) * (r["std"] + 0.05),
                    f"{r['sign_consist']:.0%}", ha="center", va="bottom" if r["mean"] >= 0 else "top",
                    fontsize=6, color="purple")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("  saved", out_path, flush=True)


plot_coef_err("orig|상대_fine24", "orig|상대_fine24",
              "원본 상대 z 0.25 (24구간) 계수 안정성",
              f"{OUT}/stability_coef_orig_rel.png")
plot_coef_err("clahe|상대_fine24", "clahe|상대_fine24",
              "CLAHE 상대 z 0.25 (24구간) 계수 안정성",
              f"{OUT}/stability_coef_clahe_rel.png")
plot_coef_err("orig|상대_coarse8", "orig|상대_coarse8",
              "원본 상대 굵은 8구간 계수 안정성",
              f"{OUT}/stability_coef_orig_rel_coarse8.png")
plot_coef_err("orig|절대16_fine", "orig|절대16_fine",
              "원본 절대 16구간 계수 안정성",
              f"{OUT}/stability_coef_orig_abs16.png")

# combined 2-panel figure for report (main)
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
for ax, key, title in [
    (axes[0], "orig|상대_fine24", "원본 상대 z (24구간)"),
    (axes[1], "orig|상대_coarse8", "원본 상대 굵은 8구간"),
]:
    C = cv_store[key]
    order = list(C.columns)
    sub = cv_df[(cv_df.tf + "|" + cv_df.set) == key].set_index("bin").loc[order].reset_index()
    x = np.arange(len(sub))
    colors = ["tab:red" if v > 0 else "tab:blue" for v in sub["mean"]]
    ax.bar(x, sub["mean"], yerr=sub["std"], color=colors, alpha=0.75,
           error_kw=dict(ecolor="k", lw=0.8, capsize=2))
    bsub = pd.DataFrame(boot_coef_rows)
    bsub = bsub[(bsub.tf + "|" + bsub.set) == key]
    if not bsub.empty:
        bsub = bsub.set_index("bin")
        for i, b in enumerate(order):
            if b in bsub.index:
                lo, hi = bsub.loc[b, "ci_lo"], bsub.loc[b, "ci_hi"]
                ax.plot([i, i], [lo, hi], color="orange", lw=1.3, zorder=5)
    ax.axhline(0, color="k", lw=0.6)
    labels = [str(b).replace("R_z", "") for b in order]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("표준화 Ridge 계수")
    ax.set_title(title + "  (검정=CV σ, 주황=boot 95% CI)")
    # mark sign-stable
    for i, r in sub.iterrows():
        mark = "●" if r["sign_consist"] >= 0.9 else ("○" if r["sign_consist"] >= 0.7 else "×")
        ax.text(i, ax.get_ylim()[1] * 0.92, mark, ha="center", fontsize=7,
                color="green" if mark == "●" else ("gray" if mark == "○" else "red"))
plt.tight_layout()
plt.savefig(f"{OUT}/stability_coef_errbars.png", dpi=130)
plt.close()
print("  saved stability_coef_errbars.png", flush=True)

# second figure: abs16 orig + clahe rel coarse
fig, axes = plt.subplots(2, 1, figsize=(11, 8))
for ax, key, title in [
    (axes[0], "clahe|상대_coarse8", "CLAHE 상대 굵은 8구간"),
    (axes[1], "orig|절대16_fine", "원본 절대 16구간"),
]:
    C = cv_store[key]
    order = list(C.columns)
    sub = cv_df[(cv_df.tf + "|" + cv_df.set) == key].set_index("bin").loc[order].reset_index()
    x = np.arange(len(sub))
    colors = ["tab:red" if v > 0 else "tab:blue" for v in sub["mean"]]
    ax.bar(x, sub["mean"], yerr=sub["std"], color=colors, alpha=0.75,
           error_kw=dict(ecolor="k", lw=0.8, capsize=2))
    bsub = pd.DataFrame(boot_coef_rows)
    bsub = bsub[(bsub.tf + "|" + bsub.set) == key]
    if not bsub.empty:
        bsub = bsub.set_index("bin")
        for i, b in enumerate(order):
            if b in bsub.index:
                lo, hi = bsub.loc[b, "ci_lo"], bsub.loc[b, "ci_hi"]
                ax.plot([i, i], [lo, hi], color="orange", lw=1.3, zorder=5)
    ax.axhline(0, color="k", lw=0.6)
    labels = [str(b).replace("R_z", "").replace("A_h16_", "") for b in order]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("표준화 Ridge 계수")
    ax.set_title(title + "  (검정=CV σ, 주황=boot 95% CI)")
plt.tight_layout()
plt.savefig(f"{OUT}/stability_coef_errbars2.png", dpi=130)
plt.close()
print("  saved stability_coef_errbars2.png", flush=True)

print("DONE", flush=True)

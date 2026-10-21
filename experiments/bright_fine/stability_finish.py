#!/usr/bin/env python3
"""Finish threshold overlap + figures + stability.md from saved CSVs."""
import os
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
n = 500

cv_df = pd.read_csv(f"{OUT}/stability_cv_coefs.csv")
boot_coef = pd.read_csv(f"{OUT}/stability_boot_coefs.csv")
boot_d = pd.read_csv(f"{OUT}/stability_boot_d.csv")
rank_df = pd.read_csv(f"{OUT}/stability_thr_ranks.csv")
thr_n = pd.read_csv(f"{OUT}/stability_thr_n.csv")

# fix overlap
overlap_rows = []
for tf in ["orig", "clahe"]:
    sets = {}
    for thr in [230, 220, 210]:
        sub = rank_df[(rank_df["tf"] == tf) & (rank_df["thr"] == thr) & (rank_df["rank"] <= 10)]
        sets[thr] = set(sub["feat"])
    for a, b in [(230, 220), (230, 210), (220, 210)]:
        inter = sets[a] & sets[b]
        overlap_rows.append({
            "tf": tf, "pair": f"{a}∩{b}", "n_overlap": len(inter),
            "feats": ", ".join(sorted(inter)),
        })
ov = pd.DataFrame(overlap_rows)
ov.to_csv(f"{OUT}/stability_thr_overlap.csv", index=False)
print(ov.to_string())

# Rebuild cv_store for plotting by re-running only CV coefs for key sets (fast)
ALPHAS = np.logspace(-1, 6, 30)

def load_feats(name):
    path = f"{OUT}/bf_feats.csv" if name == "orig" else f"{OUT}/bf_feats_{name}.csv"
    return pd.read_csv(path).iloc[:500].drop(columns="ID")

XF = {m: load_feats(m) for m in ["orig", "clahe"]}
RZ24 = [c for c in XF["orig"].columns if c.startswith("R_z") and "q" not in c]
A16 = [c for c in XF["orig"].columns if c.startswith("A_h16_")]

def merge_adjacent(X, cols, group_size, prefix):
    out = {}
    for i in range(0, len(cols), group_size):
        chunk = cols[i:i + group_size]
        a = chunk[0].replace(prefix, "")
        b = chunk[-1].replace(prefix, "")
        nm = f"{prefix}{a}~{b}" if a != b else f"{prefix}{a}"
        out[nm] = X[chunk].sum(axis=1).values
    return pd.DataFrame(out, index=X.index)

def fit_ridge_coefs(Xmat, yv):
    Z = StandardScaler().fit_transform(Xmat)
    md = RidgeCV(ALPHAS).fit(Z, yv)
    return md.coef_.copy(), float(md.alpha_)

def collect_cv(Xmat, names):
    rows = []
    for s in range(10):
        for tr, _ in KFold(5, shuffle=True, random_state=100 + s).split(np.zeros(n)):
            coef, _ = fit_ridge_coefs(Xmat[tr], y[tr])
            rows.append(pd.Series(coef, index=names))
    return pd.DataFrame(rows)

cv_store = {}
plot_specs = []
for tf in ["orig", "clahe"]:
    Xdf = XF[tf]
    M = Xdf[RZ24].values.astype(float)
    cv_store[f"{tf}|상대_fine24"] = collect_cv(M, RZ24)
    plot_specs.append((f"{tf}|상대_fine24", f"{tf} 상대 fine24"))
    cdf = merge_adjacent(Xdf, RZ24, 3, "R_z")
    cv_store[f"{tf}|상대_coarse8"] = collect_cv(cdf.values.astype(float), list(cdf.columns))
    plot_specs.append((f"{tf}|상대_coarse8", f"{tf} 상대 coarse8"))
    M16 = Xdf[A16].values.astype(float)
    cv_store[f"{tf}|절대16_fine"] = collect_cv(M16, A16)

print("cv_store rebuilt", list(cv_store.keys()))

def plot_panel(ax, key, title):
    C = cv_store[key]
    order = list(C.columns)
    sub = cv_df[(cv_df["tf"] + "|" + cv_df["set"]) == key].set_index("bin").loc[order].reset_index()
    x = np.arange(len(sub))
    colors = ["tab:red" if v > 0 else "tab:blue" for v in sub["mean"]]
    ax.bar(x, sub["mean"], yerr=sub["std"], color=colors, alpha=0.75,
           error_kw=dict(ecolor="k", lw=0.8, capsize=2))
    bsub = boot_coef[(boot_coef["tf"] + "|" + boot_coef["set"]) == key]
    if not bsub.empty:
        bsub = bsub.set_index("bin")
        for i, b in enumerate(order):
            if b in bsub.index:
                lo, hi = bsub.loc[b, "ci_lo"], bsub.loc[b, "ci_hi"]
                ax.plot([i, i], [lo, hi], color="darkorange", lw=1.4, zorder=5)
    ax.axhline(0, color="k", lw=0.6)
    labels = [str(b).replace("R_z", "").replace("A_h16_", "") for b in order]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("표준화 Ridge 계수")
    ax.set_title(title)
    # sign consistency markers
    ymax = ax.get_ylim()[1]
    for i, r in sub.iterrows():
        sc = r["sign_consist"]
        mark = "●" if sc >= 0.9 else ("○" if sc >= 0.7 else "×")
        col = "green" if mark == "●" else ("dimgray" if mark == "○" else "crimson")
        ax.text(i, ymax * 0.92, mark, ha="center", fontsize=7, color=col)

# Figure 1: orig relative fine + coarse
fig, axes = plt.subplots(2, 1, figsize=(12, 8.5))
plot_panel(axes[0], "orig|상대_fine24", "원본 상대 z 0.25 (24구간) — 검정=CV σ(50 fits), 주황=boot 95% CI, ●부호≥90%")
plot_panel(axes[1], "orig|상대_coarse8", "원본 상대 굵은 8구간 — 검정=CV σ, 주황=boot 95% CI")
plt.tight_layout()
plt.savefig(f"{OUT}/stability_coef_errbars.png", dpi=130)
plt.close()
print("saved fig1")

# Figure 2: clahe coarse8 + orig abs16
fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
plot_panel(axes[0], "clahe|상대_coarse8", "CLAHE 상대 굵은 8구간 — 검정=CV σ, 주황=boot 95% CI")
plot_panel(axes[1], "orig|절대16_fine", "원본 절대 16구간 — 검정=CV σ, 주황=boot 95% CI")
plt.tight_layout()
plt.savefig(f"{OUT}/stability_coef_errbars2.png", dpi=130)
plt.close()
print("saved fig2")

# ========== Write stability.md ==========
def fmt_coef_table(tf, sn, top_n=None, min_sign=None):
    sub = cv_df[(cv_df["tf"] == tf) & (cv_df["set"] == sn)].copy()
    b = boot_coef[(boot_coef["tf"] == tf) & (boot_coef["set"] == sn)].set_index("bin")
    sub["ci_lo"] = sub["bin"].map(b["ci_lo"] if len(b) else {})
    sub["ci_hi"] = sub["bin"].map(b["ci_hi"] if len(b) else {})
    sub["boot_sign"] = sub["bin"].map(b["sign_stable"] if len(b) else {})
    if min_sign is not None:
        sub = sub[sub["sign_consist"] >= min_sign]
    sub = sub.sort_values("mean", key=lambda s: s.abs(), ascending=False)
    if top_n:
        sub = sub.head(top_n)
    lines = ["| bin | mean | CV σ | 부호일치 | boot 95% CI | CI가 0제외 |",
             "|:----|-----:|-----:|--------:|:------------|:----------|"]
    for _, r in sub.iterrows():
        ci = ""
        excl = ""
        if pd.notna(r.get("ci_lo")):
            ci = f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
            excl = "Y" if r.get("boot_sign") else "N"
        lines.append(
            f"| {r['bin']} | {r['mean']:+.2f} | {r['std']:.2f} | {r['sign_consist']:.0%} | {ci} | {excl} |"
        )
    return "\n".join(lines)

def stable_bins(tf, sn, sign=0.9):
    sub = cv_df[(cv_df["tf"] == tf) & (cv_df["set"] == sn)]
    b = boot_coef[(boot_coef["tf"] == tf) & (boot_coef["set"] == sn)]
    if len(b):
        ok = set(b.loc[b["sign_stable"], "bin"])
        sub = sub[sub["bin"].isin(ok) & (sub["sign_consist"] >= sign)]
    else:
        sub = sub[sub["sign_consist"] >= sign]
    return sub.sort_values("mean", key=lambda s: s.abs(), ascending=False)

md = []
md.append("# 밝기 구간 특징 안정성 (bright_fine/stability)")
md.append("")
md.append("목표: 구간 면적비 Ridge 계수와 고경도 분리 효과가 표본·폴드에 따라 얼마나 흔들리는지.")
md.append("")
md.append("설정")
md.append("- Ridge: 표준화 후 RidgeCV(`logspace(-1,6,30)`), **면적비만**(베이스 특징 없음).")
md.append("- CV: 5-fold × 10회(seed `100+s`) → 계수 세트 50개.")
md.append("- Bootstrap: train 500장 1000회 재표집(계수), 고경도/대조군 층화 1000회(효과크기 d).")
md.append("- 고경도 경계: >230 / >220 / >210. 대조군은 `[170,230] ∩ 비고경도`.")
md.append("- 굵은 구간: 상대 24→8(인접 3개 합), 24→6(인접 4개 합); 절대16→8(인접 2개 합).")
md.append("")
md.append("그림: `stability_coef_errbars.png`(원본 상대 fine/coarse), `stability_coef_errbars2.png`(CLAHE 상대 coarse + 원본 절대16).")
md.append("")

# summary counts
md.append("## 1) 교차검증 계수 안정성 요약")
md.append("")
md.append("부호일치 ≥90% 이고(해당 시) bootstrap 95% CI가 0을 안 지나는 구간:")
md.append("")
for tf, sn in [("orig", "상대_fine24"), ("orig", "상대_coarse8"), ("orig", "상대_coarse6"),
               ("orig", "절대16_fine"), ("clahe", "상대_fine24"), ("clahe", "상대_coarse8"),
               ("clahe", "절대16_fine")]:
    st = stable_bins(tf, sn, 0.9)
    names = ", ".join(f"{r.bin}({r['mean']:+.2f})" for _, r in st.iterrows()) or "(없음)"
    md.append(f"- **{tf} / {sn}**: {names}")
md.append("")

md.append("### 원본 상대 fine24 (상위 |계수| + 안정성)")
md.append("")
md.append(fmt_coef_table("orig", "상대_fine24", top_n=12))
md.append("")
md.append("### 원본 상대 coarse8")
md.append("")
md.append(fmt_coef_table("orig", "상대_coarse8"))
md.append("")
md.append("### CLAHE 상대 coarse8")
md.append("")
md.append(fmt_coef_table("clahe", "상대_coarse8"))
md.append("")
md.append("### 원본 절대16")
md.append("")
md.append(fmt_coef_table("orig", "절대16_fine", top_n=12))
md.append("")

# sign consistency stats
md.append("### 세트별 부호일치 분포")
md.append("")
md.append("| tf | set | bins | 부호≥90% | 부호≥70% | 평균 부호일치 |")
md.append("|:---|:----|-----:|--------:|--------:|-------------:|")
for (tf, sn), g in cv_df.groupby(["tf", "set"], sort=False):
    md.append(
        f"| {tf} | {sn} | {len(g)} | {(g.sign_consist>=0.9).sum()} | "
        f"{(g.sign_consist>=0.7).sum()} | {g.sign_consist.mean():.0%} |"
    )
md.append("")

md.append("## 2) Bootstrap 효과크기 d (고경도 >230 vs 대조 170–230)")
md.append("")
md.append("층화 부트스트랩 1000회. CI가 0을 포함하면 표본 운에 민감.")
md.append("")
for tf in ["orig", "clahe"]:
    md.append(f"### {tf}")
    md.append("")
    md.append("| rank | feat | type | d | 95% CI | 0제외 |")
    md.append("|-----:|:-----|:-----|----:|:-------|:-----|")
    sub = boot_d[boot_d["tf"] == tf].sort_values("rank")
    for _, r in sub.iterrows():
        md.append(
            f"| {int(r['rank'])} | {r['feat']} | {r['type']} | {r['d']:+.3f} | "
            f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {'Y' if r['ci_excludes_0'] else 'N'} |"
        )
    md.append("")

md.append("## 3) 고경도 경계 민감도")
md.append("")
md.append("| 기준 | n_hi | n_ctl(170–230∩비hi) |")
md.append("|-----:|-----:|--------------------:|")
for _, r in thr_n.iterrows():
    md.append(f"| >{int(r.thr)} | {int(r.n_hi)} | {int(r.n_ctl)} |")
md.append("")
md.append("상위 10 특징 겹침:")
md.append("")
md.append("| tf | 쌍 | 겹침 수 | 특징 |")
md.append("|:---|:---|-------:|:-----|")
for _, r in ov.iterrows():
    md.append(f"| {r.tf} | {r.pair} | {int(r.n_overlap)} | {r.feats} |")
md.append("")
md.append("경계별 상위 5 (원본):")
md.append("")
md.append("| thr | rank | feat | d |")
md.append("|----:|-----:|:-----|----:|")
for thr in [230, 220, 210]:
    sub = rank_df[(rank_df["tf"] == "orig") & (rank_df["thr"] == thr) & (rank_df["rank"] <= 5)]
    for _, r in sub.sort_values("rank").iterrows():
        md.append(f"| >{int(thr)} | {int(r['rank'])} | {r['feat']} | {r['d']:+.3f} |")
md.append("")

# conclusions
md.append("## 4) 결론 — 표본을 바꿔도 남는 신호")
md.append("")

# compute concrete stable findings
st_o8 = stable_bins("orig", "상대_coarse8", 0.9)
st_c8 = stable_bins("clahe", "상대_coarse8", 0.9)
st_of = stable_bins("orig", "상대_fine24", 0.9)
st_abs = stable_bins("orig", "절대16_fine", 0.9)

# d features with CI excluding 0
d_stable = boot_d[boot_d["ci_excludes_0"] == True]

md.append("1. **면적비 Ridge 계수(세분 24구간)는 대체로 불안정.** 인접 구간 상관이 커서 부호가 폴드마다 뒤집히는 구간이 많음. "
          f"원본 상대 fine24에서 부호일치≥90%는 {len(st_of)}개뿐.")
md.append("")
md.append("2. **굵게 합치면(8·6구간) 안정성이 크게 올라감.** "
          f"원본 상대 coarse8 안정 구간: {', '.join(st_o8['bin'].tolist()) or '없음'}. "
          f"CLAHE coarse8: {', '.join(st_c8['bin'].tolist()) or '없음'}. "
          "패턴은 이전 보고와 같음 — **중저(대략 z −1.5~−0.75 부근)와 중고(대략 z +1~+1.75)는 경도↑, "
          "중간(z ≈ −0.5~+0.25)과 아주 밝음(z ≳ +2)은 경도↓.**")
md.append("")
md.append("3. **절대 밝기 구간은 상대보다 약하고 덜 안정적.** "
          f"원본 절대16에서 부호·CI 동시 안정: {', '.join(st_abs['bin'].tolist()) or '거의 없음'}.")
md.append("")
n_d_ok = int(d_stable[d_stable.tf == "orig"].shape[0])
md.append(f"4. **고경도 분리(d)의 상위 신호는 면적비보다 연결성분(덩어리 수↓·크기↑).** "
          f"원본 상위10 중 bootstrap CI가 0을 제외하는 것: {n_d_ok}/10. "
          "대표: `R_cc[-1.5,-1)_maxA`(큼), `R_cc[0.5,1)_n`·`R_cc[-1,-0.5)_n`·`R_cc[-1.5,-1)_n`(적음).")
md.append("")
ov230_220 = int(ov[(ov.tf == "orig") & (ov.pair == "230∩220")].n_overlap.iloc[0])
ov230_210 = int(ov[(ov.tf == "orig") & (ov.pair == "230∩210")].n_overlap.iloc[0])
md.append(f"5. **경계를 220·210으로 낮춰 표본을 늘려도 상위 특징 족보는 유지.** "
          f"원본 top10 겹침: 230∩220={ov230_220}, 230∩210={ov230_210}. "
          "다만 |d|는 경계가 낮을수록 대체로 줄어듦(고경도 정의가 희석).")
md.append("")
md.append("6. **실무 함의:** 세분 구간별 가중치를 그대로 해석·보정에 쓰기엔 표본 운이 큼. "
          "쓸 만하면 **굵은 상대 밝기 대역(6~8구간)** 과 **덩어리(연결성분) 특징** 정도. "
          "그래도 최고 블렌드(12.60)에 얹으면 이득은 이전 평가처럼 노이즈 수준.")
md.append("")
md.append("산출물: `stability_cv_coefs.csv`, `stability_boot_coefs.csv`, `stability_boot_d.csv`, "
          "`stability_thr_ranks.csv`, `stability_thr_overlap.csv`, "
          "`stability_coef_errbars.png`, `stability_coef_errbars2.png`.")

path = f"{OUT}/stability.md"
with open(path, "w") as f:
    f.write("\n".join(md) + "\n")
print("wrote", path)
print("--- preview ---")
print("\n".join(md[:80]))

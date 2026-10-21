#!/usr/bin/env python3
import os
os.chdir("./runs")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

OUT = "bright_fine"
y = pd.read_csv("./open/train.csv").hardness.values
n = 500
cv_df = pd.read_csv(f"{OUT}/stability_cv_coefs.csv")
boot_coef = pd.read_csv(f"{OUT}/stability_boot_coefs.csv")
ALPHAS = np.logspace(-1, 6, 30)

def load(name):
    p = f"{OUT}/bf_feats.csv" if name == "orig" else f"{OUT}/bf_feats_{name}.csv"
    return pd.read_csv(p).iloc[:500].drop(columns="ID")

XF = {m: load(m) for m in ["orig", "clahe"]}
RZ24 = [c for c in XF["orig"].columns if c.startswith("R_z") and "q" not in c]
A16 = [c for c in XF["orig"].columns if c.startswith("A_h16_")]

def merge_adj(X, cols, g, prefix):
    out = {}
    for i in range(0, len(cols), g):
        chunk = cols[i:i+g]
        a, b = chunk[0].replace(prefix, ""), chunk[-1].replace(prefix, "")
        out[f"{prefix}{a}~{b}" if a != b else f"{prefix}{a}"] = X[chunk].sum(1).values
    return pd.DataFrame(out)

def fit(X, yv):
    return RidgeCV(ALPHAS).fit(StandardScaler().fit_transform(X), yv).coef_

def collect(Xmat, names):
    rows = []
    for s in range(10):
        for tr, _ in KFold(5, shuffle=True, random_state=100+s).split(np.zeros(n)):
            rows.append(pd.Series(fit(Xmat[tr], y[tr]), index=names))
    return pd.DataFrame(rows)

cv_store = {}
for tf in ["orig", "clahe"]:
    X = XF[tf]
    cv_store[f"{tf}|상대_fine24"] = collect(X[RZ24].values.astype(float), RZ24)
    c8 = merge_adj(X, RZ24, 3, "R_z")
    cv_store[f"{tf}|상대_coarse8"] = collect(c8.values.astype(float), list(c8.columns))
    cv_store[f"{tf}|절대16_fine"] = collect(X[A16].values.astype(float), A16)

def panel(ax, key, title):
    C = cv_store[key]
    order = list(C.columns)
    sub = cv_df[(cv_df.tf + "|" + cv_df.set) == key].set_index("bin").loc[order].reset_index()
    x = np.arange(len(sub))
    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in sub["mean"]]
    ax.bar(x, sub["mean"], yerr=sub["std"], color=colors, alpha=0.8,
           error_kw=dict(ecolor="k", lw=0.8, capsize=2))
    bsub = boot_coef[(boot_coef.tf + "|" + boot_coef.set) == key]
    if len(bsub):
        bsub = bsub.set_index("bin")
        for i, b in enumerate(order):
            if b in bsub.index:
                lo, hi = bsub.loc[b, "ci_lo"], bsub.loc[b, "ci_hi"]
                ax.plot([i, i], [lo, hi], color="#dd8452", lw=1.5, zorder=5)
    ax.axhline(0, color="k", lw=0.6)
    labels = [str(b).replace("R_z", "z").replace("A_h16_", "") for b in order]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("Std. Ridge coef")
    ax.set_title(title, fontsize=10)
    ymax = max(ax.get_ylim()[1], 0.1)
    for i, r in sub.iterrows():
        sc = r["sign_consist"]
        mark = "*" if sc >= 0.9 else ("+" if sc >= 0.7 else "x")
        col = "green" if mark == "*" else ("gray" if mark == "+" else "red")
        ax.text(i, ymax * 0.92, mark, ha="center", fontsize=8, color=col,
                transform=ax.get_xaxis_transform() and ax.transData or ax.transData)
    # simpler: use axes fraction for markers at top
    ymin, ymax = ax.get_ylim()
    for i, r in sub.iterrows():
        sc = r["sign_consist"]
        mark = "*" if sc >= 0.9 else ("." if sc >= 0.7 else "x")
        col = "green" if mark == "*" else ("0.4" if mark == "." else "crimson")
        ax.text(i, ymax - 0.04 * (ymax - ymin), mark, ha="center", fontsize=8, color=col)

fig, axes = plt.subplots(2, 1, figsize=(12, 8.5))
panel(axes[0], "orig|상대_fine24",
      "orig relative z (24 bins)  |  black=CV sd (50 fits), orange=boot 95% CI  |  *=sign>=90%")
panel(axes[1], "orig|상대_coarse8",
      "orig relative coarse (8 bins)  |  black=CV sd, orange=boot 95% CI")
fig.suptitle("Brightness bin Ridge coef stability", fontsize=12, y=1.0)
plt.tight_layout()
plt.savefig(f"{OUT}/stability_coef_errbars.png", dpi=130, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
panel(axes[0], "clahe|상대_coarse8",
      "CLAHE relative coarse (8 bins)  |  black=CV sd, orange=boot 95% CI")
panel(axes[1], "orig|절대16_fine",
      "orig absolute 16 bins  |  black=CV sd, orange=boot 95% CI")
fig.suptitle("Brightness bin Ridge coef stability (cont.)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/stability_coef_errbars2.png", dpi=130, bbox_inches="tight")
plt.close()
print("figs ok")

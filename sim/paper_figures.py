"""Generate the paper figures from the unified 100-trial JSONs."""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
F = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(F, exist_ok=True)

def load(name):
    return json.load(open(os.path.join(R, name)))

def err(ax, x, y, lo, hi, color, mk, label):
    y = np.array(y); lo = np.array(lo); hi = np.array(hi)
    ax.errorbar(x, y, yerr=[y-lo, hi-y], fmt=mk+"-", color=color,
                capsize=2, lw=1.2, ms=4, label=label)

# Fig. 1: CRLB ratio and block CRLB
kr = load("kappa_real.json"); cm = load("crlb_multicomponent.json")
cl = load("crlb_lfm.json")
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
aa = [r["alpha"] for r in kr]
ax[0].plot(aa, [r["kappa"]/0.5 for r in kr], "ko-", label=r"real $\kappa(2)/\kappa(\alpha)$")
aa2 = [r["alpha"] for r in cl["single"]]
ax[0].plot(aa2, [r["rf"] for r in cl["single"]], "s--", label=r"complex $J(2)/J(\alpha)$")
ax[0].set_xlabel(r"$\alpha$"); ax[0].set_ylabel("CRLB ratio vs Gaussian")
ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
xs = [r["df"] for r in cm["multi"]]
ax[1].semilogy(xs, [r["cf"] for r in cm["multi"]], "o-", label="block CRLB variance")
ax[1].axhline(cm["single"]["cf"], color="k", ls=":", label="single-component")
ax[1].set_xlabel(r"$\Delta f\,N$"); ax[1].set_ylabel("CRLB of $f_1$")
ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig1_crlb.pdf"))

# Fig. 2: single-component estimators
us = load("unified_single.json")
fig, ax = plt.subplots(figsize=(6.8, 3.7))
aa = [r["alpha"] for r in us]
for key, lab, col, mk in [("ls","LS","C0","o"),("hu","Huber","C1","s"),("ml","MLE","C2","^")]:
    err(ax, aa, [r[key]["r"] for r in us], [r[key]["r_lo"] for r in us],
        [r[key]["r_hi"] for r in us], col, mk, lab)
ax.axhline(0.6745, color="k", ls=":", lw=1, label="Gaussian 0.6745")
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("MAD of $f$ / CRLB std")
ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig2_estimators.pdf"))

# Fig. 3: FrFT
fig, ax = plt.subplots(figsize=(6.8, 3.7))
for key, lab, col, mk in [("fl","FrFT+LS","C0","o"),("fc","FrFT only","C3","s")]:
    err(ax, aa, [r[key]["r"] for r in us], [r[key]["r_lo"] for r in us],
        [r[key]["r_hi"] for r in us], col, mk, lab)
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("MAD of $f$ / CRLB std")
ax.set_yscale("log"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig3_frft.pdf"))

# Fig. 4: multicomponent vs separation
md = load("unified_multi_df.json")
fig, ax = plt.subplots(figsize=(6.8, 3.7))
xs = [r["df"] for r in md]
for key, lab, col, mk in [("s_ls","single LS","C0","o"),("t_ls","two LS","C0","^"),
                          ("s_hu","single Huber","C1","s"),("t_hu","two Huber","C1","v")]:
    err(ax, xs, [r[key]["mad"] for r in md], [r[key]["lo"] for r in md],
        [r[key]["hi"] for r in md], col, mk, lab)
ax.set_xlabel(r"$\Delta f\,N$"); ax.set_ylabel("MAD of $f_1$ / CRLB std")
ax.set_yscale("log"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig4_multicomponent.pdf"))

# Fig. 5: alpha sweep at sub-Rayleigh
ma = load("unified_multi_alpha.json")
fig, ax = plt.subplots(figsize=(6.8, 3.7))
aa = [r["alpha"] for r in ma]
for key, lab, col, mk in [("s_ls","single LS","C0","o"),("t_ls","two LS","C0","^"),
                          ("s_hu","single Huber","C1","s"),("t_hu","two Huber","C1","v")]:
    err(ax, aa, [r[key]["mad"] for r in ma], [r[key]["lo"] for r in ma],
        [r[key]["hi"] for r in ma], col, mk, lab)
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("MAD of $f_1$ / CRLB std")
ax.set_yscale("log"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig5_alpha.pdf"))

# Fig. 6: GELCT surrogate
gb = load("gelct_baseline.json")
fig, ax = plt.subplots(figsize=(6.8, 3.7))
aa = [r["alpha"] for r in gb]
for key, lab, col, mk in [("ls","LS","C0","o"),("hu","Huber","C1","s"),
                          ("ml","MLE","C2","^"),("gel_log","GELCT-like","C3","v")]:
    ax.plot(aa, [r[key] for r in gb], mk+"-", color=col, ms=4, label=lab)
ax.axhline(0.6745, color="k", ls=":", lw=1, label="Gaussian 0.6745")
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("MAD of $f$ / CRLB std")
ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig6_gelct.pdf"))
print("wrote figures to", F)

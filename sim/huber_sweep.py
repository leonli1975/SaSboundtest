"""Validity boundary: Huber delta sweep vs alpha, with LS/MLE references."""
import json, os, sys, numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "..", "paper1", "code"))
from estimators_sas_fast import ls_fit, sig, F0, K0, SCALE
from estimators_mle_mad import mle_fit, crlb_std, code_density
from m1_quant_corrected import csas

RES = os.path.join(HERE, "..", "results")

def huber_delta(x, N, f0, k0, fac, iters=4):
    f, k = f0, k0
    e = sig(N, f, k); r = x - np.vdot(e, x)/N*e
    for _ in range(iters):
        def wobj(p):
            f, k = p; e = sig(N, f, k)
            r = x - np.vdot(e, x)/N*e; rr = np.abs(r)
            delta = fac*np.median(rr)
            return float(np.sum(np.where(rr <= delta, 0.5*rr**2, delta*(rr - 0.5*delta))))
        res = minimize(wobj, [f, k], method="Nelder-Mead",
                       options={"xatol": 1e-11, "fatol": 1e-14, "maxiter": 250})
        f, k = res.x
        e = sig(N, f, k); r = x - np.vdot(e, x)/N*e; rr = np.abs(r)
        delta = fac*np.median(rr)
        w = np.minimum(1.0, delta/np.maximum(rr, 1e-12))
        c = np.sum(w*np.conj(e)*x)/np.sum(w); r = x - c*e
    return f, k

if __name__ == "__main__":
    rng = np.random.default_rng(20261105)
    N = 256
    alphas = [1.3, 1.5, 1.7, 2.0]
    factors = [0.5, 0.75, 1.0, 1.345, 1.75, 2.0, 2.5, 3.0]
    trials = 60
    rows = []
    for a in alphas:
        rc, hc, phi, J, c, r50 = code_density(a)
        logh = np.log(np.maximum(hc, 1e-300))
        cf, ck = crlb_std(a, N)
        ls_f, ml_f = [], []
        errs = {fac: [] for fac in factors}
        for t in range(trials):
            x = sig(N, F0, K0) + SCALE*csas(N, a, rng)
            fl, kl = ls_fit(x, N); ls_f.append(abs(fl-F0))
            fm, km = mle_fit(x, N, rc, logh); ml_f.append(abs(fm-F0))
            for fac in factors:
                fh, kh = huber_delta(x, N, fl, kl, fac)
                errs[fac].append(abs(fh-F0))
        ls_r = np.median(ls_f)/cf
        ml_r = np.median(ml_f)/cf
        curve = {fac: np.median(errs[fac])/cf for fac in factors}
        best_fac = min(curve, key=curve.get)
        rows.append(dict(alpha=a, cf=cf, ls=ls_r, mle=ml_r,
                         curve=curve, best_fac=best_fac, best=curve[best_fac]))
        print(f"\n=== alpha={a:.2f}  LS={ls_r:.2f}  MLE={ml_r:.2f}")
        print("  delta factor -> MAD/CRLB: " +
              "  ".join(f"{fac:g}:{curve[fac]:.2f}" for fac in factors))
        print(f"  best delta factor={best_fac:g}  MAD/CRLB={curve[best_fac]:.2f}")
    json.dump(rows, open(os.path.join(RES, "huber_sweep.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for a, mk in zip(alphas, ["o","s","^","D"]):
        r = next(x for x in rows if x["alpha"] == a)
        ys = [r["curve"][fac] for fac in factors]
        ax.plot(factors, ys, mk+"-", label=f"alpha={a}")
        ax.plot([r["best_fac"]], [r["best"]], "k*", ms=11)
    ax.axhline(0.6745, color="k", ls=":", lw=1, label="Gaussian 0.6745")
    ax.set_xlabel("Huber delta factor (x median residual)")
    ax.set_ylabel("MAD of f / CRLB std")
    ax.set_title("Huber tuning vs alpha (N=256); stars = best factor")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "huber_sweep.png"), dpi=170)
    print("\nwrote results/huber_sweep.json and huber_sweep.png")

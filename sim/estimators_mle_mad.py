"""MLE/Huber/LS MAD vs CRLB (robust metric; MSE is outlier-dominated)."""
import json, os, sys, numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "..", "paper1", "code"))
from estimators_sas_fast import ls_fit, huber_fit, sig, F0, K0, SCALE
from crlb_lfm import gram_single, crlb, Jmap
from fisher_clip import code_density
from m1_quant_corrected import csas

RES = os.path.join(HERE, "..", "results")

def mle_obj(p, x, N, rc, logh):
    f, k, A, phi = p
    n = np.arange(N)
    e = A*np.exp(1j*(2*np.pi*f*n + np.pi*k*n*n + phi))
    r = np.abs(x - e)
    return -float(np.sum(np.interp(r, rc, logh, left=logh[0], right=logh[-1])))

def mle_fit(x, N, rc, logh):
    fl, kl = ls_fit(x, N)
    e = sig(N, fl, kl); c = np.vdot(e, x)/N
    res = minimize(mle_obj, [fl, kl, abs(c), np.angle(c)], args=(x, N, rc, logh),
                   method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 350})
    return res.x[0], res.x[1]

def crlb_std(alpha, N):
    G = gram_single(N, F0, K0)
    C, _ = crlb(G, Jmap[alpha], [0, 1])
    return np.sqrt(C[0, 0])*SCALE, np.sqrt(C[1, 1])*SCALE

if __name__ == "__main__":
    rng = np.random.default_rng(20261103)
    N = 256
    alphas = [1.3, 1.5, 1.7, 2.0]
    trials = 50
    rows = []
    for a in alphas:
        rc, hc, phi, J, c, r50 = code_density(a)
        logh = np.log(np.maximum(hc, 1e-300))
        cf, ck = crlb_std(a, N)
        ef_ls, ek_ls, ef_hu, ek_hu, ef_ml, ek_ml = [], [], [], [], [], []
        for t in range(trials):
            x = sig(N, F0, K0) + SCALE*csas(N, a, rng)
            fl, kl = ls_fit(x, N)
            fh, kh = huber_fit(x, N, fl, kl)
            fm, km = mle_fit(x, N, rc, logh)
            ef_ls.append(abs(fl-F0)); ek_ls.append(abs(kl-K0))
            ef_hu.append(abs(fh-F0)); ek_hu.append(abs(kh-K0))
            ef_ml.append(abs(fm-F0)); ek_ml.append(abs(km-K0))
        mfl, mkl = np.median(ef_ls), np.median(ek_ls)
        mfh, mkh = np.median(ef_hu), np.median(ek_hu)
        mfm, mkm = np.median(ef_ml), np.median(ek_ml)
        rows.append(dict(alpha=a, cf=cf, ck=ck,
                         ls_f=mfl/cf, ls_k=mkl/ck,
                         hu_f=mfh/cf, hu_k=mkh/ck,
                         ml_f=mfm/cf, ml_k=mkm/ck))
        print(f"\n=== alpha={a:.2f}  CRLB std f={cf:.3e} k={ck:.3e}")
        print(f"  MAD/CRLB  LS f={mfl/cf:.2f} k={mkl/ck:.2f} | "
              f"Huber f={mfh/cf:.2f} k={mkh/ck:.2f} | "
              f"MLE f={mfm/cf:.2f} k={mkm/ck:.2f}")
    json.dump(rows, open(os.path.join(RES, "estimators_mle_mad.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    aa = [r["alpha"] for r in rows]
    ax.plot(aa, [r["ls_f"] for r in rows], "o-", label="LS")
    ax.plot(aa, [r["hu_f"] for r in rows], "s--", label="Huber")
    ax.plot(aa, [r["ml_f"] for r in rows], "^-", label="MLE")
    ax.axhline(0.6745, color="k", ls=":", lw=1, label="Gaussian MAD/CRLB")
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("MAD of $f$ / CRLB std")
    ax.set_title("LFM estimation (N=256): LS vs Huber vs MLE")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "estimators_mle_mad.png"), dpi=170)
    print("\nwrote results/estimators_mle_mad.json and estimators_mle_mad.png")

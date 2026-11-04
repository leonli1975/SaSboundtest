"""FrFT/chirp-search coarse + LS/Huber refine vs the CRLB."""
import json, os, sys, numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "..", "paper1", "code"))
from estimators_mle_mad import (mle_fit, crlb_std, F0, K0, SCALE, sig,
                                huber_fit, csas, code_density)
from estimators_sas_fast import ls_obj, ls_fit

RES = os.path.join(HERE, "..", "results")

def ls_fit_at(x, N, f0, k0):
    res = minimize(ls_obj, [f0, k0], args=(x, N), method="Nelder-Mead",
                   options={"xatol": 1e-11, "fatol": 1e-14, "maxiter": 300})
    return res.x

def frft_coarse(x, N, nk=201):
    """FrFT-equivalent chirp search: dechirp + DFT, peak over (k, f)."""
    n = np.arange(N)
    kg = np.linspace(-1.0/N, 1.0/N, nk)
    best = (-1.0, 0.0, 0)
    for k in kg:
        Y = np.fft.fft(x*np.exp(-1j*np.pi*k*n*n))
        idx = int(np.argmax(np.abs(Y)))
        val = float(np.abs(Y[idx]))
        if val > best[0]:
            best = (val, k, idx)
    f = best[2]/N
    if f > 0.5: f -= 1.0
    return f, best[1]

if __name__ == "__main__":
    rng = np.random.default_rng(20261104)
    N = 256
    alphas = [1.3, 1.5, 1.7, 2.0]
    trials = 50
    rows = []
    for a in alphas:
        rc, hc, phi, J, c, r50 = code_density(a)
        logh = np.log(np.maximum(hc, 1e-300))
        cf, ck = crlb_std(a, N)
        e_ls, e_hu, e_ml, e_fl, e_fh, e_fc = [], [], [], [], [], []
        for t in range(trials):
            x = sig(N, F0, K0) + SCALE*csas(N, a, rng)
            fl, kl = ls_fit(x, N)
            fh, kh = huber_fit(x, N, fl, kl)
            fm, km = mle_fit(x, N, rc, logh)
            ff, kf = frft_coarse(x, N)
            flf, klf = ls_fit_at(x, N, ff, kf)
            fhf, khf = huber_fit(x, N, flf, klf)
            e_ls.append(abs(fl-F0)); e_hu.append(abs(fh-F0)); e_ml.append(abs(fm-F0))
            e_fl.append(abs(flf-F0)); e_fh.append(abs(fhf-F0)); e_fc.append(abs(ff-F0))
        row = dict(alpha=a, cf=cf,
                   ls=np.median(e_ls)/cf, hu=np.median(e_hu)/cf,
                   ml=np.median(e_ml)/cf, fl=np.median(e_fl)/cf,
                   fh=np.median(e_fh)/cf, fc=np.median(e_fc)/cf)
        rows.append(row)
        print(f"\n=== alpha={a:.2f}  CRLB std f={cf:.3e}")
        print(f"  MAD/CRLB f: LS={row['ls']:.2f}  Huber={row['hu']:.2f}  MLE={row['ml']:.2f}  "
              f"FrFT+LS={row['fl']:.2f}  FrFT+Huber={row['fh']:.2f}  FrFT only={row['fc']:.2f}")
    json.dump(rows, open(os.path.join(RES, "estimators_frft.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    aa = [r["alpha"] for r in rows]
    ax.plot(aa, [r["ls"] for r in rows], "o-", label="LS")
    ax.plot(aa, [r["hu"] for r in rows], "s--", label="Huber")
    ax.plot(aa, [r["ml"] for r in rows], "^-", label="MLE")
    ax.plot(aa, [r["fl"] for r in rows], "d--", label="FrFT+LS")
    ax.plot(aa, [r["fh"] for r in rows], "v--", label="FrFT+Huber")
    ax.axhline(0.6745, color="k", ls=":", lw=1, label="Gaussian MAD/CRLB")
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("MAD of $f$ / CRLB std")
    ax.set_title("LFM estimation (N=256): LS/Huber/MLE/FrFT")
    ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "estimators_frft.png"), dpi=170)
    print("\nwrote results/estimators_frft.json and estimators_frft.png")

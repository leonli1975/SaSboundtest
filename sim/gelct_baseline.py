"""GELCT-like baseline (nonlinear amplitude + chirp search + LS) vs LS/Huber/MLE."""
import json, os, sys, numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "..", "paper1", "code"))
from estimators_sas_fast import ls_fit, huber_fit, sig, F0, K0, SCALE, ls_obj
from estimators_mle_mad import mle_fit, crlb_std, code_density
from m1_quant_corrected import csas

RES = os.path.join(HERE, "..", "results")

def chirp_search(y, N, nk=201):
    n = np.arange(N); kg = np.linspace(-1.0/N, 1.0/N, nk)
    best = (-1.0, 0.0, 0)
    for k in kg:
        Y = np.fft.fft(y*np.exp(-1j*np.pi*k*n*n))
        idx = int(np.argmax(np.abs(Y))); val = float(np.abs(Y[idx]))
        if val > best[0]: best = (val, k, idx)
    f = best[2]/N
    if f > 0.5: f -= 1.0
    return f, best[1]

def ls_fit_at(x, N, f0, k0):
    res = minimize(ls_obj, [f0, k0], args=(x, N), method="Nelder-Mead",
                   options={"xatol": 1e-11, "fatol": 1e-14, "maxiter": 250})
    return res.x

def gelct_like(x, N, mode="log", rfac=1.0):
    r = np.abs(x); R = max(rfac*np.median(r), 1e-6); ph = x/np.maximum(r, 1e-12)
    amp = np.log1p(r/R) if mode == "log" else r/(1.0 + r/R)
    y = amp*ph
    f, k = chirp_search(y, N)
    return ls_fit_at(x, N, f, k)

def glct(x, N):
    f, k = chirp_search(x, N)
    return ls_fit_at(x, N, f, k)

if __name__ == "__main__":
    rng = np.random.default_rng(20261109)
    N = 256
    alphas = [1.3, 1.5, 1.7, 2.0]
    trials = 50
    rows = []
    for a in alphas:
        rc, hc, phi, J, c, r50 = code_density(a)
        logh = np.log(np.maximum(hc, 1e-300))
        cf, ck = crlb_std(a, N)
        e = {k: [] for k in ["ls", "hu", "ml", "glct", "gel_log", "gel_sig"]}
        for t in range(trials):
            x = sig(N, F0, K0) + SCALE*csas(N, a, rng)
            fl, kl = ls_fit(x, N); e["ls"].append(abs(fl-F0))
            fh, kh = huber_fit(x, N, fl, kl); e["hu"].append(abs(fh-F0))
            fm, km = mle_fit(x, N, rc, logh); e["ml"].append(abs(fm-F0))
            fg, kg = glct(x, N); e["glct"].append(abs(fg-F0))
            fgl, kgl = gelct_like(x, N, "log"); e["gel_log"].append(abs(fgl-F0))
            fgs, kgs = gelct_like(x, N, "sig"); e["gel_sig"].append(abs(fgs-F0))
        row = dict(alpha=a, cf=cf, **{k: np.median(v)/cf for k, v in e.items()})
        rows.append(row)
        print(f"\n=== alpha={a:.2f}  CRLB std f={cf:.3e}")
        print("  MAD/CRLB f: " + "  ".join(f"{k}={row[k]:.2f}" for k in ["ls","hu","ml","glct","gel_log","gel_sig"]))
    json.dump(rows, open(os.path.join(RES, "gelct_baseline.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    aa = [r["alpha"] for r in rows]
    styles = {"ls":"o-","hu":"s--","ml":"^-","glct":"d--","gel_log":"v-","gel_sig":"x-"}
    labels = {"ls":"LS","hu":"Huber","ml":"MLE","glct":"GLCT (linear)",
              "gel_log":"GELCT-like log","gel_sig":"GELCT-like sigmoid"}
    for k, st in styles.items():
        ax.plot(aa, [r[k] for r in rows], st, label=labels[k])
    ax.axhline(0.6745, color="k", ls=":", lw=1, label="Gaussian 0.6745")
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("MAD of f / CRLB std")
    ax.set_title("GELCT-like baseline vs LS/Huber/MLE (N=256)")
    ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "gelct_baseline.png"), dpi=170)
    print("\nwrote results/gelct_baseline.json and gelct_baseline.png")

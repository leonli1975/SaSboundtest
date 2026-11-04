"""Fast estimator efficiency: LS and Huber vs the S-alpha-S CRLB."""
import json, os, sys, numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "..", "paper1", "code"))
from crlb_lfm import gram_single, crlb, Jmap
from m1_quant_corrected import csas

RES = os.path.join(HERE, "..", "results")
F0, K0 = 0.1, 0.001
SCALE = 0.5

def sig(N, f, k):
    n = np.arange(N)
    return np.exp(1j*(2*np.pi*f*n + np.pi*k*n*n))

def ls_obj(p, x, N):
    f, k = p; e = sig(N, f, k)
    c = np.vdot(e, x)/N; r = x - c*e
    return float(np.real(np.vdot(r, r)))

def ls_fit(x, N):
    res = minimize(ls_obj, [F0, K0], args=(x, N), method="Nelder-Mead",
                   options={"xatol": 1e-11, "fatol": 1e-14, "maxiter": 300})
    return res.x

def huber_fit(x, N, f0, k0, iters=4):
    f, k = f0, k0
    e = sig(N, f, k); r = x - np.vdot(e, x)/N*e
    delta = 1.345*np.median(np.abs(r))
    for _ in range(iters):
        def wobj(p):
            f, k = p; e = sig(N, f, k)
            r = x - np.vdot(e, x)/N*e; rr = np.abs(r)
            return float(np.sum(np.where(rr <= delta, 0.5*rr**2, delta*(rr - 0.5*delta))))
        res = minimize(wobj, [f, k], method="Nelder-Mead",
                       options={"xatol": 1e-11, "fatol": 1e-14, "maxiter": 250})
        f, k = res.x; e = sig(N, f, k)
        r = x - np.vdot(e, x)/N*e; rr = np.abs(r)
        delta = 1.345*np.median(rr)
        w = np.minimum(1.0, delta/np.maximum(rr, 1e-12))
        c = np.sum(w*np.conj(e)*x)/np.sum(w); r = x - c*e
    return f, k

def crlb_std(alpha, N):
    G = gram_single(N, F0, K0)
    C, _ = crlb(G, Jmap[alpha], [0, 1])
    return np.sqrt(C[0, 0])*SCALE, np.sqrt(C[1, 1])*SCALE

if __name__ == "__main__":
    rng = np.random.default_rng(20261101)
    alphas = [1.3, 1.5, 1.7, 2.0]
    Ns = [128, 256, 512]
    trials = 50
    rows = []
    for a in alphas:
        cf, ck = crlb_std(a, 256)
        print(f"\n=== alpha={a:.2f} CRLB std (N=256): f={cf:.3e} k={ck:.3e}")
        for N in Ns:
            cfN, ckN = crlb_std(a, N)
            ef_ls, ek_ls, ef_hu, ek_hu = [], [], [], []
            for t in range(trials):
                x = sig(N, F0, K0) + SCALE*csas(N, a, rng)
                fl, kl = ls_fit(x, N)
                fh, kh = huber_fit(x, N, fl, kl)
                ef_ls.append(abs(fl-F0)); ek_ls.append(abs(kl-K0))
                ef_hu.append(abs(fh-F0)); ek_hu.append(abs(kh-K0))
            mfl, mkl = np.median(ef_ls), np.median(ek_ls)
            mfh, mkh = np.median(ef_hu), np.median(ek_hu)
            rows.append(dict(alpha=a, N=N, cf=cfN, ck=ckN,
                             mfl=mfl, mkl=mkl, mfh=mfh, mkh=mkh,
                             rfl=mfl/cfN, rkl=mkl/ckN, rfh=mfh/cfN, rkh=mkh/ckN))
            print(f" N={N:4d} LS f={mfl/cfN:.2f} k={mkl/ckN:.2f} | Huber f={mfh/cfN:.2f} k={mkh/ckN:.2f} (MAD/CRLB)")
    json.dump(rows, open(os.path.join(RES, "estimators_sas_fast.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for a, mk in zip(alphas, ["o","s","^","D"]):
        rr = [r for r in rows if r["alpha"] == a]
        ax[0].loglog([r["N"] for r in rr], [r["mfl"] for r in rr], mk+"-", label=f"LS a={a}")
        ax[0].loglog([r["N"] for r in rr], [r["mfh"] for r in rr], mk+"--", alpha=.6)
        ax[0].loglog([r["N"] for r in rr], [r["cf"] for r in rr], "k:", lw=1)
    ax[0].set_xlabel("N"); ax[0].set_ylabel("MAD of f"); ax[0].set_title("LS (solid) vs Huber (dashed); CRLB dotted")
    ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=6)
    for a, mk in zip(alphas, ["o","s","^","D"]):
        rr = [r for r in rows if r["alpha"] == a]
        ax[1].loglog([r["N"] for r in rr], [r["mkl"] for r in rr], mk+"-", label=f"LS a={a}")
        ax[1].loglog([r["N"] for r in rr], [r["mkh"] for r in rr], mk+"--", alpha=.6)
        ax[1].loglog([r["N"] for r in rr], [r["ck"] for r in rr], "k:", lw=1)
    ax[1].set_xlabel("N"); ax[1].set_ylabel("MAD of k"); ax[1].set_title("LS (solid) vs Huber (dashed); CRLB dotted")
    ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "estimators_sas_fast.png"), dpi=170)
    print("\nwrote results/estimators_sas_fast.json and estimators_sas_fast.png")

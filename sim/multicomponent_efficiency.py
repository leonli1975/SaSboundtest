"""Two-component LFM: single-component vs two-component estimators and the block CRLB."""
import json, os, sys, numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "..", "paper1", "code"))
from crlb_lfm import gram_single, gram_multi, crlb, Jmap
from m1_quant_corrected import csas

RES = os.path.join(HERE, "..", "results")
F1, K1 = 0.1, 0.001
SCALE = 0.5

def sig(N, f, k):
    n = np.arange(N)
    return np.exp(1j*(2*np.pi*f*n + np.pi*k*n*n))

def ls_single_obj(p, x, N):
    f, k = p; e = sig(N, f, k)
    c = np.vdot(e, x)/N; r = x - c*e
    return float(np.real(np.vdot(r, r)))

def ls_single(x, N, f0, k0):
    res = minimize(ls_single_obj, [f0, k0], args=(x, N), method="Nelder-Mead",
                   options={"xatol": 1e-11, "fatol": 1e-14, "maxiter": 300})
    return res.x

def amps_two(e1, e2, x):
    A = np.array([[np.vdot(e1, e1), np.vdot(e1, e2)],
                  [np.vdot(e2, e1), np.vdot(e2, e2)]], dtype=complex)
    b = np.array([np.vdot(e1, x), np.vdot(e2, x)], dtype=complex)
    return np.linalg.solve(A, b)

def ls_two_obj(p, x, N):
    f1, k1, f2, k2 = p
    e1 = sig(N, f1, k1); e2 = sig(N, f2, k2)
    c = amps_two(e1, e2, x)
    r = x - c[0]*e1 - c[1]*e2
    return float(np.real(np.vdot(r, r)))

def ls_two(x, N, f0, k0, f2, k2):
    res = minimize(ls_two_obj, [f0, k0, f2, k2], args=(x, N), method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 500})
    return res.x

def huber_single(x, N, f0, k0, fac=1.345, iters=4):
    f, k = f0, k0
    e = sig(N, f, k); r = x - np.vdot(e, x)/N*e
    for _ in range(iters):
        def wobj(p):
            f, k = p; e = sig(N, f, k)
            r = x - np.vdot(e, x)/N*e; rr = np.abs(r)
            d = fac*np.median(rr)
            return float(np.sum(np.where(rr <= d, 0.5*rr**2, d*(rr - 0.5*d))))
        res = minimize(wobj, [f, k], method="Nelder-Mead",
                       options={"xatol": 1e-11, "fatol": 1e-14, "maxiter": 250})
        f, k = res.x
        e = sig(N, f, k); r = x - np.vdot(e, x)/N*e; rr = np.abs(r)
        d = fac*np.median(rr); w = np.minimum(1.0, d/np.maximum(rr, 1e-12))
        c = np.sum(w*np.conj(e)*x)/np.sum(w); r = x - c*e
    return f, k

def huber_two(x, N, f0, k0, f2, k2, fac=1.345, iters=4):
    f1, kk1, ff2, kk2 = f0, k0, f2, k2
    e1 = sig(N, f1, kk1); e2 = sig(N, ff2, kk2)
    c = amps_two(e1, e2, x); r = x - c[0]*e1 - c[1]*e2
    for _ in range(iters):
        def wobj(p):
            f1, kk1, ff2, kk2 = p
            e1 = sig(N, f1, kk1); e2 = sig(N, ff2, kk2)
            c = amps_two(e1, e2, x); r = x - c[0]*e1 - c[1]*e2
            rr = np.abs(r); d = fac*np.median(rr)
            return float(np.sum(np.where(rr <= d, 0.5*rr**2, d*(rr - 0.5*d))))
        res = minimize(wobj, [f1, kk1, ff2, kk2], method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 400})
        f1, kk1, ff2, kk2 = res.x
        e1 = sig(N, f1, kk1); e2 = sig(N, ff2, kk2)
        c = amps_two(e1, e2, x); r = x - c[0]*e1 - c[1]*e2
        rr = np.abs(r); d = fac*np.median(rr)
        w = np.minimum(1.0, d/np.maximum(rr, 1e-12))
        A = np.array([[np.sum(w*np.abs(e1)**2), np.sum(w*np.conj(e1)*e2)],
                      [np.sum(w*np.conj(e2)*e1), np.sum(w*np.abs(e2)**2)]], dtype=complex)
        b = np.array([np.sum(w*np.conj(e1)*x), np.sum(w*np.conj(e2)*x)], dtype=complex)
        c = np.linalg.solve(A, b); r = x - c[0]*e1 - c[1]*e2
    return f1, kk1, ff2, kk2

if __name__ == "__main__":
    rng = np.random.default_rng(20261107)
    N = 256
    alpha = 1.5
    dfs = [0.5, 1.0, 2.0, 4.0, 8.0]
    trials = 50
    rows = []
    for df in dfs:
        f2 = F1 + df/N
        # CRLB
        G1 = gram_single(N, F1, K1)
        C1, _ = crlb(G1, Jmap[alpha], [0, 1])
        Gm = gram_multi(N, [(F1, K1, 1.0, 0.0), (f2, K1, 1.0, 0.0)])
        C2, _ = crlb(Gm, Jmap[alpha], [0, 1])
        e_s, e_t, e_hs, e_ht = [], [], [], []
        for t in range(trials):
            x = sig(N, F1, K1) + sig(N, f2, K1) + SCALE*csas(N, alpha, rng)
            fs, ks = ls_single(x, N, F1, K1)
            ft = ls_two(x, N, F1, K1, f2, K1)
            fhs, khs = huber_single(x, N, F1, K1)
            fht, kht, _, _ = huber_two(x, N, F1, K1, f2, K1)
            e_s.append(abs(fs-F1)); e_t.append(abs(ft[0]-F1))
            e_hs.append(abs(fhs-F1)); e_ht.append(abs(fht-F1))
        rows.append(dict(df=df, cf1=C1[0,0], cf2=C2[0,0],
                         single_ls=np.median(e_s), two_ls=np.median(e_t),
                         single_hu=np.median(e_hs), two_hu=np.median(e_ht),
                         single_ls_r=np.median(e_s)/np.sqrt(C1[0,0]),
                         two_ls_r=np.median(e_t)/np.sqrt(C2[0,0]),
                         single_hu_r=np.median(e_hs)/np.sqrt(C1[0,0]),
                         two_hu_r=np.median(e_ht)/np.sqrt(C2[0,0])))
        print(f"\ndf={df:.1f}/N  singleCRLB={np.sqrt(C1[0,0]):.2e} twoCRLB={np.sqrt(C2[0,0]):.2e}")
        print(f"  single LS MAD/CRLB={rows[-1]['single_ls_r']:.2f}  two LS={rows[-1]['two_ls_r']:.2f}")
        print(f"  single Huber={rows[-1]['single_hu_r']:.2f}  two Huber={rows[-1]['two_hu_r']:.2f}")
    json.dump(rows, open(os.path.join(RES, "multicomponent_efficiency.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
    xs = [r["df"] for r in rows]
    ax[0].plot(xs, [r["single_ls_r"] for r in rows], "o-", label="single-component LS")
    ax[0].plot(xs, [r["two_ls_r"] for r in rows], "s--", label="two-component LS")
    ax[0].plot(xs, [r["single_hu_r"] for r in rows], "^-", label="single-component Huber")
    ax[0].plot(xs, [r["two_hu_r"] for r in rows], "v--", label="two-component Huber")
    ax[0].axhline(0.6745, color="k", ls=":", lw=1, label="Gaussian 0.6745")
    ax[0].set_xlabel(r"separation $\Delta f\,N$"); ax[0].set_ylabel("MAD / CRLB")
    ax[0].set_title("Single vs two-component estimators"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
    ax[1].semilogy(xs, [r["cf1"] for r in rows], "o-", label="single-component CRLB")
    ax[1].semilogy(xs, [r["cf2"] for r in rows], "s--", label="two-component CRLB")
    ax[1].set_xlabel(r"separation $\Delta f\,N$"); ax[1].set_ylabel("CRLB of $f_1$")
    ax[1].set_title("Block CRLB vs single-component CRLB"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "multicomponent_efficiency.png"), dpi=170)
    print("\nwrote results/multicomponent_efficiency.json and .png")

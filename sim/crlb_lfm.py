"""CRLB for LFM parameters under S-alpha-S with nuisance A,phi and multi-component."""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
D = json.load(open(os.path.join(RES, "fisher_clip.json")))
Jmap = {round(r["alpha"], 2): r["J"] for r in D["raw"]}
J2 = Jmap[2.0]

def gram_single(N, f, k, A=1.0, phi=0.0):
    n = np.arange(N)
    s = A*np.exp(1j*(2*np.pi*f*n + np.pi*k*n*n + phi))
    u = [1j*2*np.pi*n*s, 1j*np.pi*n*n*s, s/A, 1j*s]   # f,k,A,phi
    G = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            G[i, j] = np.real(np.sum(np.conj(u[i])*u[j]))
    return G

def gram_multi(N, comps):
    """comps: list of (f,k,A,phi)."""
    n = np.arange(N)
    s = sum(A*np.exp(1j*(2*np.pi*f*n + np.pi*k*n*n + phi)) for f, k, A, phi in comps)
    u = []
    for f, k, A, phi in comps:
        si = A*np.exp(1j*(2*np.pi*f*n + np.pi*k*n*n + phi))
        u += [1j*2*np.pi*n*si, 1j*np.pi*n*n*si, si/A, 1j*si]
    m = len(u); G = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            G[i, j] = np.real(np.sum(np.conj(u[i])*u[j]))
    return G

def crlb(G, J, idx):
    I = 0.5*J*G
    C = np.linalg.inv(I)
    return C[np.ix_(idx, idx)], C

if __name__ == "__main__":
    N = 256
    # ---------- single component ----------
    G = gram_single(N, 0.1, 0.001)
    idx = [0, 1]
    C2, _ = crlb(G, J2, idx)              # Gaussian reference
    print("Single-component Gaussian CRLB (f,k):", C2[0,0], C2[1,1])
    rows = []
    for a in [1.10, 1.30, 1.50, 1.70, 1.90, 2.00]:
        C, _ = crlb(G, Jmap[a], idx)
        rows.append(dict(alpha=a, J=Jmap[a], cf=C[0,0], ck=C[1,1],
                         rf=C[0,0]/C2[0,0], rk=C[1,1]/C2[1,1]))
        print(f"alpha={a:.2f}: J={Jmap[a]:.4f}  CRLB_f={C[0,0]:.6e}  CRLB_k={C[1,1]:.6e}  "
              f"ratio_f={C[0,0]/C2[0,0]:.4f} ratio_k={C[1,1]/C2[1,1]:.4f}")

    # Aldimashki Gaussian check (absolute factor; model-dependent)
    ald = 90.0/(1.0*np.pi**2*N**5)
    print(f"\nAldimashki 90 sigma^2/(A^2 pi^2 T^5) with sigma=1,A=1,T=N: {ald:.6e}")
    print(f"computed Gaussian CRLB_k: {C2[1,1]:.6e}; factor computed/Aldimashki: {C2[1,1]/ald:.4f}")

    # ---------- multi-component ----------
    print("\nMulti-component CRLB (f1) vs separation (alpha=1.5 vs 2.0):")
    mrows = []
    for df in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        comps = [(0.1, 0.001, 1.0, 0.0),
                 (0.1 + df/N, 0.001, 1.0, 0.0)]
        Gm = gram_multi(N, comps)
        C15, _ = crlb(Gm, Jmap[1.5], [0, 1])
        C20, _ = crlb(Gm, Jmap[2.0], [0, 1])
        cond = np.linalg.cond(Gm)
        mrows.append(dict(df=df, cf15=C15[0,0], cf20=C20[0,0],
                          ratio=C15[0,0]/C20[0,0], cond=cond))
        print(f"df={df:5.1f}/N  CRLB_f(1.5)={C15[0,0]:.6e}  CRLB_f(2.0)={C20[0,0]:.6e}  "
              f"ratio={C15[0,0]/C20[0,0]:.4f}  cond(G)={cond:.3e}")

    json.dump(dict(single=rows, multi=mrows, aldimashki=ald,
                   gaussian_ck=C2[1,1], gaussian_cf=C2[0,0]),
              open(os.path.join(RES, "crlb_lfm.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5))
    aa = [r["alpha"] for r in rows]
    ax[0].plot(aa, [r["rf"] for r in rows], "o-", label="CRLB f ratio")
    ax[0].plot(aa, [r["rk"] for r in rows], "s--", label="CRLB k ratio")
    ax[0].axhline(1, color="k", ls=":", lw=1)
    ax[0].set_xlabel(r"$\alpha$"); ax[0].set_ylabel("CRLB / Gaussian")
    ax[0].set_title("Single-component CRLB penalty"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    xs = [r["df"] for r in mrows]
    ax[1].semilogy(xs, [r["cf15"] for r in mrows], "o-", label=r"$\alpha$=1.5")
    ax[1].semilogy(xs, [r["cf20"] for r in mrows], "s--", label=r"$\alpha$=2.0")
    ax[1].set_xlabel(r"frequency separation $\Delta f\,N$")
    ax[1].set_ylabel("CRLB of $f_1$"); ax[1].set_title("Two-component resolution (alpha scales only)")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "crlb_lfm.png"), dpi=170)
    print("\nwrote results/crlb_lfm.json and results/crlb_lfm.png")

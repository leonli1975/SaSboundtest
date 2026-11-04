"""Real SαS Fisher factor kappa(alpha) and multicomponent CRLB (stable)."""
import json, os, sys, numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.append(HERE)
from crlb_lfm import gram_single, gram_multi, crlb, Jmap, J2

RES = os.path.join(HERE, "..", "results")

def real_density(alpha, x):
    return (1/np.pi)*quad(lambda t: np.exp(-t**alpha)*np.cos(x*t),
                          0, 80, limit=300, epsabs=1e-9, epsrel=1e-7)[0]

def kappa(alpha, n=2001, xmax=12.0):
    if alpha >= 1.999:
        return 0.5   # analytic Gaussian N(0,2) Fisher factor
    x = np.linspace(-xmax, xmax, n)
    f = np.array([real_density(alpha, xi) for xi in x])
    f = np.maximum(f, 1e-12)
    fp = np.gradient(f, x)
    return float(np.trapezoid(fp*fp/np.maximum(f, 1e-300), x))

if __name__ == "__main__":
    alphas = [1.0, 1.3, 1.5, 1.7, 1.9, 2.0]
    rows = []
    for a in alphas:
        k = kappa(a)
        rows.append(dict(alpha=a, kappa=k))
        print(f"alpha={a:.2f}  kappa={k:.4f}")
    k2 = next(r['kappa'] for r in rows if r['alpha'] == 2.0)
    print("\nReal SαS vs complex Fisher ratios:")
    for r in rows:
        a = r['alpha']; k = r['kappa']
        if a in Jmap:
            print(f"  alpha={a:.2f}  real kappa(2)/kappa(a)={k2/k:.4f}  "
                  f"complex J(2)/J(a)={J2/Jmap[a]:.4f}")
    json.dump(rows, open(os.path.join(RES, "kappa_real.json"), "w"), indent=1)

    # multicomponent CRLB at alpha=1.5
    N = 256
    G1 = gram_single(N, 0.1, 0.001)
    C1, _ = crlb(G1, Jmap[1.5], [0, 1])
    print("\nSingle-component CRLB(f,k) at alpha=1.5:", C1[0,0], C1[1,1])
    mrows = []
    print("\nTwo-component CRLB(f1) vs separation:")
    for df in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        Gm = gram_multi(N, [(0.1, 0.001, 1.0, 0.0), (0.1+df/N, 0.001, 1.0, 0.0)])
        C, _ = crlb(Gm, Jmap[1.5], [0, 1])
        cond = np.linalg.cond(Gm)
        mrows.append(dict(df=df, cf=C[0,0], ck=C[1,1], cond=cond))
        print(f"  df={df:5.1f}/N  CRLB_f1={C[0,0]:.4e}  ratio_to_single={C[0,0]/C1[0,0]:.2f}  cond={cond:.2e}")
    json.dump(dict(single=dict(cf=C1[0,0], ck=C1[1,1]), multi=mrows),
              open(os.path.join(RES, "crlb_multicomponent.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
    common = [a for a in alphas if a in Jmap]
    ax[0].plot(common, [next(r['kappa'] for r in rows if r['alpha']==a)/k2 for a in common],
               'ko-', label='real SαS kappa/kappa(2)')
    ax[0].plot(common, [J2/Jmap[a] for a in common], 's--', label='complex J(2)/J(a)')
    ax[0].set_xlabel(r'$\alpha$'); ax[0].set_ylabel('Fisher ratio vs Gaussian')
    ax[0].set_title('Real vs complex SαS Fisher factor'); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    xs = [r['df'] for r in mrows]
    ax[1].semilogy(xs, [r['cf'] for r in mrows], 'o-', label='two-component CRLB(f1)')
    ax[1].axhline(C1[0,0], color='k', ls=':', label='single-component CRLB(f)')
    ax[1].set_xlabel(r'frequency separation $\Delta f\,N$')
    ax[1].set_ylabel('CRLB of $f_1$'); ax[1].set_title('Multicomponent CRLB vs separation')
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "kappa_multicomponent.png"), dpi=170)
    print("\nwrote results/kappa_real.json, crlb_multicomponent.json, kappa_multicomponent.png")

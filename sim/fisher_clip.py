"""A1/A2: S-alpha-S Fisher information for LFM parameters; injective vs non-injective clipping."""
import numpy as np, json, os
from scipy.integrate import quad
from scipy.special import j0
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MED = 0.8325546
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

def cf_density(alpha, rg):
    h = np.empty_like(rg)
    for i, r in enumerate(rg):
        h[i] = (1/(2*np.pi))*quad(lambda t: j0(r*t)*np.exp(-t**alpha)*t,
                                  0, 80, limit=400, epsabs=1e-9, epsrel=1e-7)[0]
    return h

def code_density(alpha, n=2400, rmax=25.0):
    rg = np.linspace(0.005, rmax, n)
    h = gaussian_filter1d(cf_density(alpha, rg), 2)
    pdf = 2*np.pi*rg*h
    cdf = np.cumsum(pdf)*(rg[1]-rg[0]); cdf /= cdf[-1]
    r50 = float(np.interp(0.5, cdf, rg)); c = r50/MED
    rc = rg/c; hc = c*c*h
    phi = np.gradient(hc, rc)/(hc+1e-300)     # signed radial score
    J = float(np.trapezoid(phi*phi*hc*2*np.pi*rc, rc))
    return rc, hc, phi, J, c, r50

def hard_clip_loss(rc, hc, phi, J, R0):
    """Y=min(R,R0): J_Y = E[E[phi|Y]^2] = J - P(R>R0) Var(phi|R>R0)."""
    m = rc > R0
    p = float(np.trapezoid(hc[m]*2*np.pi*rc[m], rc[m]))
    if p <= 0:
        return 0.0, 0.0
    mu = float(np.trapezoid(phi[m]*hc[m]*2*np.pi*rc[m], rc[m])/p)
    var = float(np.trapezoid((phi[m]-mu)**2*hc[m]*2*np.pi*rc[m], rc[m])/p)
    loss = p*var/J
    return float(loss), p

if __name__ == "__main__":
    alphas = [1.10, 1.30, 1.50, 1.70, 1.90, 2.00]
    rows = []
    for a in alphas:
        rc, hc, phi, J, c, r50 = code_density(a)
        rows.append(dict(alpha=a, J=J, c=c, r50=r50))
        print(f"alpha={a:.2f}  r50={r50:.4f}  c={c:.4f}  J_code={J:.4f}")
    J2 = next(r['J'] for r in rows if r['alpha'] == 2.0)
    print("\nFisher factor / CRLB penalty:")
    for r in rows:
        print(f"  alpha={r['alpha']:.2f}: J/J2={r['J']/J2:.4f}  CRLB={J2/r['J']:.3f}x")

    print("\nHard-clip loss  P(R>R0)*Var(phi|R>R0)/J  (non-injective):")
    clip = []
    for a in [1.10, 1.50, 1.90, 2.00]:
        rc, hc, phi, J, c, r50 = code_density(a)
        for R0 in [1.0, 2.0, 3.0, 5.0]:
            L, p = hard_clip_loss(rc, hc, phi, J, R0)
            clip.append(dict(alpha=a, R0=R0, loss=L, mass=p))
            print(f"  alpha={a:.2f} R0={R0:.0f}: loss={L:.4f} mass={p:.4f}")
    print("\nInjective maps (PNAT r^p, sigmoid r/(1+r/R)): loss = 0 by the")
    print("Fisher-information data-processing inequality (one-to-one map is sufficient).")

    json.dump(dict(raw=rows, hard_clip=clip,
                   note="Injective clipping has zero Fisher-information loss; only non-injective maps lose."),
              open(os.path.join(OUT, "fisher_clip.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
    aa = [r['alpha'] for r in rows]; jj = [r['J']/J2 for r in rows]
    ax[0].plot(aa, jj, 'ko-'); ax[0].set_xlabel(r'$\alpha$'); ax[0].set_ylabel(r'$J(\alpha)/J(2)$')
    ax[0].set_title('Fisher factor for location/LFM parameters'); ax[0].grid(alpha=.3)
    for a in [1.5, 1.9, 2.0]:
        ys = [r['loss'] for r in clip if r['alpha']==a]
        xs = [r['R0'] for r in clip if r['alpha']==a]
        ax[1].plot(xs, ys, 'o-', label=r'$\alpha$=%.1f'%a)
    ax[1].axhline(0, color='k', ls='--', lw=1, label='injective (PNAT/sigmoid)')
    ax[1].set_xlabel(r'hard-clip level $R_0$'); ax[1].set_ylabel('information loss')
    ax[1].set_title('Only non-injective maps lose'); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fisher_clip.png"), dpi=170)
    print("\nwrote results/fisher_clip.json and results/fisher_clip.png")

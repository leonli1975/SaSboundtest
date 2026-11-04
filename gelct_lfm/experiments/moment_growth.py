# -*- coding: utf-8 -*-
"""
Why the two estimators cannot behave the same way -- a moment argument.

Equation (8) is a pointwise, phase-preserving change of the amplitude scale:

    f[r] = (log_e(|r|^(1/e)) + 1)/|r| * r      =>      |f[r]| = |1 + ln|r| / e|

For a symmetric alpha-stable variable X:
  * P(|X| > t) ~ C t^(-alpha), so P(ln|X| > s) ~ C exp(-alpha s): ln|X| has
    *exponential* tails and therefore moments of every order.
      => all moments of f[X] are finite.
  * X itself has no finite moment of order >= alpha, so the second moment is
    infinite for every alpha < 2, and the empirical second moment of a sample
    of size n grows like n^(2/alpha - 1) without bound.

This script plots the running empirical moment as a function of the sample
size.  A method that minimises a quadratic loss (LS) scales with the raw
signal, so its effective "noise power" is this diverging curve; a method that
runs on f[r] sees a curve that settles down.  That is the structural reason the
two cannot collapse together.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sas_noise import sas_rvs
from gelct_core import nonlinear_transform, E

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

NSAMP = 400_000
alphas = [0.8, 1.0, 1.2, 1.5, 2.0]
rng = np.random.default_rng(424242)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.0))
cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(alphas)))

print(f"{'alpha':>6}{'E|X|^2 (n=4e5)':>20}{'E|f[X]|^2 (n=4e5)':>22}   n^(2/alpha-1)")
for c, a in zip(cols, alphas):
    X = sas_rvs(a, 1.0, NSAMP, rng)
    run_raw = np.cumsum(np.abs(X) ** 2) / np.arange(1, NSAMP + 1)
    fx = np.abs(nonlinear_transform(X.astype(complex)))
    run_tx = np.cumsum(fx ** 2) / np.arange(1, NSAMP + 1)
    n = np.arange(1, NSAMP + 1)
    step = 200
    ax[0].loglog(n[::step], run_raw[::step], color=c, lw=1.4, label=f"$\\alpha$={a}")
    ax[1].loglog(n[::step], run_tx[::step], color=c, lw=1.4, label=f"$\\alpha$={a}")
    print(f"{a:>6}{run_raw[-1]:>20.3e}{run_tx[-1]:>22.6f}   {NSAMP ** (2.0/a - 1.0):.3e}")

# generic 2/alpha - 1 reference slopes on the raw panel
n = np.arange(1, NSAMP + 1)
for a, c in zip([0.8, 1.5], ["crimson", "k"]):
    ax[0].loglog(n[::step], 2.0 * (n[::step] ** (2.0 / a - 1.0)) / (n[::step] ** (2.0 / a - 1.0))[0],
                 ls=":", color=c, lw=1.2, label=f"$n^{{2/\\alpha-1}}$, $\\alpha$={a}")

ax[0].set_xlabel("number of samples  n")
ax[0].set_ylabel(r"running $\frac{1}{n}\sum |X_i|^2$")
ax[0].set_title(r"(a) raw S$\alpha$S samples: the 2nd moment never converges")
ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("number of samples  n")
ax[1].set_ylabel(r"running $\frac{1}{n}\sum |f[X_i]|^2$")
ax[1].set_title(r"(b) after Eq. (8): every moment is finite and converges")
ax[1].set_ylim(0.3, None)
ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8)

fig.suptitle("The mechanism: Eq. (8) turns an infinite-variance sequence into a finite-moment one", fontsize=13)
fig.tight_layout()
png = os.path.join(OUT, "collapse_moments.png")
fig.savefig(png, dpi=130)
print("saved:", png)

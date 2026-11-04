# -*- coding: utf-8 -*-
"""
Step 2 of the controlled comparison: the same chirplet transform, with and
without the Eq. (8) nonlinear transform.

Both panels use the *identical* transform code (Eq. 7), the identical window,
hop and FFT length; the only difference is whether f[r] or the raw r enters.
A "ridge concentration" index quantifies how much energy sits inside +-1.5
frequency bins of the true instantaneous frequency.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gelct_core import (LFMParams, received_signal, nonlinear_transform, gelct,
                        GELCTConfig)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

N, FS, TS = 1024, 10240.0, 0.1
P = LFMParams(1.0, 1000.0, 10000.0)
CFG = GELCTConfig(win_len=256, hop=8, nfft=1024, L=32)


class RawWrapper:
    """same transform, but the input is fed in untouched."""
    def __init__(self, r):
        self.r = r


def gelct_raw(r, fs, ts, cfg):
    """Eq. (7) applied to the raw signal: this is the plain (linear) chirplet
    transform -- exactly what Long et al. (2022) report degenerating under
    alpha-stable noise."""
    import gelct_core as gc
    orig = gc.nonlinear_transform
    gc.nonlinear_transform = lambda x: np.asarray(x, dtype=complex)
    try:
        return gelct(r, fs, ts, cfg)
    finally:
        gc.nonlinear_transform = orig


def concentration(G, tf, k_true, f0_true):
    T, F = np.meshgrid(tf.t_axis, tf.f_axis, indexing="ij")
    band = np.abs((F - (f0_true + k_true * T)) / tf.df) <= 1.5
    A = np.abs(G)
    return float(A[band].sum() ** 2 / (A ** 2).sum() / band.sum())


fig, ax = plt.subplots(2, 2, figsize=(13, 8))
print(f"{'alpha':>6}{'GSNR':>7}{'LCT(raw)':>12}{'GELCT(f[r])':>14}")
for col, alpha in enumerate([1.0, 1.5]):
    rng = np.random.default_rng(31337 + int(alpha * 10))
    r, _ = received_signal(N, FS, P, alpha=alpha, gsnr_db=-5.0, rng=rng)
    tfr = gelct_raw(r, FS, TS, CFG)
    tfg = gelct(r, FS, TS, CFG)
    cr = concentration(tfr.G, tfr, P.chirp_rate, P.f0)
    cg = concentration(tfg.G, tfg, P.chirp_rate, P.f0)
    print(f"{alpha:>6}{-5.0:>7.1f}{cr:>12.4f}{cg:>14.4f}")
    t = np.arange(N) / FS
    for row, (tf, ttl, c) in enumerate([
            (tfr, f"LCT of the raw signal (no Eq. 8),  $\\alpha$={alpha}",
             cr),
            (tfg, f"GELCT of f[r] (Eq. 7 + Eq. 8),  $\\alpha$={alpha}", cg)]):
        a = ax[row, col]
        im = a.imshow(20 * np.log10(np.abs(tf.G).T + 1e-9), origin="lower",
                      aspect="auto",
                      extent=[tf.t_axis[0] * 1e3, tf.t_axis[-1] * 1e3,
                              0, tf.f_axis[-1]],
                      cmap="viridis", vmin=-60, vmax=0)
        a.plot(t * 1e3, P.f0 + P.chirp_rate * t, "r--", lw=1.2, label="true IF")
        a.set_xlabel("time [ms]"); a.set_ylabel("freq [Hz]")
        a.set_title(ttl + f"\nridge concentration = {c:.4f}", fontsize=10)
        a.legend(loc="upper left", fontsize=8)
        plt.colorbar(im, ax=a, label="dB")
fig.suptitle("GSNR = -5 dB:  the Eq. (8) transform is what keeps the ridge alive "
             "(identical transform code in both rows)", fontsize=13)
fig.tight_layout()
png = os.path.join(OUT, "collapse_lct_vs_gelct.png")
fig.savefig(png, dpi=130)
print("saved:", png)

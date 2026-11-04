# -*- coding: utf-8 -*-
"""
Single-run demonstration of the GELCT-based LFM parameter estimator.

Produces  results/demo_single.png   with
    (a) the nonlinear transform f[.] of Eq. (8)  - amplitude compression
    (b) the received signal in S-alpha-S noise
    (c) GELCT time-frequency plane  vs  plain STFT of f[r]
    (d) the Radon transform R(u,v) of Eq. (12) and the detected line
and prints the final estimates together with the CRLBs.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sas_noise import kappa_alpha, crlb_chirp_rate, crlb_initial_frequency
from gelct_core import (LFMParams, lfm_signal, received_signal, nonlinear_transform,
                        gelct, GELCTConfig, radon_sinogram, preprocess_tf,
                        estimate_lfm_params, estimate_initial_frequency,
                        estimate_chirp_rate, default_k_max, refine_k_coherent)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------
N, fs, Ts = 1024, 10240.0, 0.1
ALPHA, GSNR = 1.5, -5.0
p = LFMParams(amplitude=1.0, f0=1000.0, chirp_rate=10000.0)
cfg = GELCTConfig(win_len=256, hop=8, nfft=1024, L=32)

rng = np.random.default_rng(20211101)
r, comp = received_signal(N, fs, p, alpha=ALPHA, gsnr_db=GSNR, rng=rng)

# ---------------- (a) the nonlinear transform ----------------------------
amp = np.logspace(-6, 6, 400)
gain = np.abs(nonlinear_transform(amp + 0j))

# ---------------- (c) GELCT vs plain STFT --------------------------------
tf = gelct(r, fs, Ts, cfg)
tf_stft = gelct(r, fs, Ts, GELCTConfig(win_len=256, hop=8, nfft=1024, L=1,
                                       use_chirplet=False))

# ---------------- (d) Radon sinogram ------------------------------------
k_hat, rad = estimate_chirp_rate(tf)
mag = preprocess_tf(tf.G, power=4.0, prep="white")
R, v_axis, u_axis = radon_sinogram(mag, n_angles=361)

# ---------------- estimates ---------------------------------------------
res = estimate_lfm_params(r, fs, Ts, cfg)
f0_hat = res["f0_hat"]
k_hat = res["k_hat"]
k_ref = refine_k_coherent(r, fs, k_hat, span=0.05)
f0_ref, _ = estimate_initial_frequency(r, k_ref, fs)

# concentration of the ridge: fraction of the plane energy that lies inside a
# +/-1 bin neighbourhood of the true instantaneous frequency
Tg, Fg = np.meshgrid(tf.t_axis, tf.f_axis, indexing="ij")
resid = (Fg - (p.f0 + p.chirp_rate * Tg)) / tf.df
band = np.abs(resid) <= 1.5
conc = lambda M: float(np.abs(M)[band].sum() ** 2 / (np.abs(M) ** 2).sum() / band.sum())
print(f"ridge concentration  STFT = {conc(tf_stft.G):.4f}   GELCT = {conc(tf.G):.4f}")

kap = kappa_alpha(ALPHA)
gsnr_lin = 10 ** (GSNR / 10.0)
cr_k = np.sqrt(crlb_chirp_rate(N, 1.0/FS, kap, gsnr_lin))
cr_f = np.sqrt(crlb_initial_frequency(N, 1.0/FS, kap, gsnr_lin))

print("=" * 70)
print(f"alpha = {ALPHA},  GSNR = {GSNR} dB,  signal amplitude A = {comp['amplitude']:.4f}")
print(f"true      : k = {p.chirp_rate:12.2f}   f0 = {p.f0:10.2f}")
print(f"estimated : k = {k_hat:12.2f}   f0 = {f0_hat:10.2f}   "
      f"(rel.err {abs(k_hat-p.chirp_rate)/p.chirp_rate:.2e}, "
      f"{abs(f0_hat-p.f0)/p.f0:.2e})")
print(f"refined   : k = {k_ref:12.2f}   f0 = {f0_ref:10.2f}   "
      f"(coherent full-window Radon limit; rel.err "
      f"{abs(k_ref-p.chirp_rate)/p.chirp_rate:.2e})")
print(f"rocket angle v_hat = {rad['v_hat']:.6f} rad   ->  -cot(v)*df/dt = "
      f"{-np.cos(rad['v_hat'])/np.sin(rad['v_hat'])*tf.df/tf.dt:.2f}")
print(f"CRLB std : k >= {cr_k:.4e}   f0 >= {cr_f:.4e}   "
      f"(kappa({ALPHA}) = {kap:.6f})")
print("=" * 70)

# ---------------- figure -------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

ax[0, 0].loglog(amp, gain, lw=2, color="tab:blue")
ax[0, 0].axhline(1.0, ls="--", c="grey", lw=1)
ax[0, 0].set_xlabel("|r|")
ax[0, 0].set_ylabel("|f[r]|")
ax[0, 0].set_title("(a) Eq.(8) nonlinear transform: amplitude compression")
ax[0, 0].grid(True, which="both", alpha=0.3)
ax[0, 0].text(2e-6, 3.0, r"$|f[r]|=|\ln|r|/e+1|$", fontsize=11)

t = np.arange(N) / fs
ax[0, 1].plot(t * 1e3, np.real(r), lw=0.6, color="tab:red")
ax[0, 1].plot(t * 1e3, np.real(comp["s"]), lw=1.4, color="tab:blue", label="LFM interference")
ax[0, 1].set_ylim(-5, 5)
ax[0, 1].set_xlabel("time [ms]")
ax[0, 1].set_title(f"(b) received signal, real part  ($\\alpha$={ALPHA}, GSNR={GSNR} dB)")
ax[0, 1].legend(loc="upper right", fontsize=8)

for a, (M, ttl) in zip(ax[1], [(np.abs(tf_stft.G), "(c1) plain STFT of f[r]"),
                               (np.abs(tf.G), "(c2) GELCT of f[r]  (Eq.7)")]):
    im = a.imshow(20 * np.log10(M.T + 1e-6), origin="lower", aspect="auto",
                  extent=[tf.t_axis[0] * 1e3, tf.t_axis[-1] * 1e3, 0, tf.f_axis[-1]],
                  cmap="viridis", vmin=-40, vmax=0)
    a.plot(t * 1e3, (p.f0 + p.chirp_rate * t), "r--", lw=1.2, label="true IF")
    a.set_xlabel("time [ms]"); a.set_ylabel("frequency [Hz]")
    a.set_title(ttl); a.legend(loc="upper left", fontsize=8)
    plt.colorbar(im, ax=a, label="dB")

fig.suptitle("GELCT: nonlinear transform + generalized extended linear chirplet transform",
             fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "demo_single_tf.png"), dpi=130)

fig2, axs = plt.subplots(1, 2, figsize=(13, 4.6))
axs[0].imshow(R, origin="lower", aspect="auto", cmap="magma",
              extent=[np.degrees(v_axis[0]), np.degrees(v_axis[-1]),
                      u_axis[0], u_axis[-1]])
axs[0].axvline(np.degrees(rad["v_hat"]), color="cyan", ls="--", lw=1.4,
               label=r"$\hat v$ from the peak")
axs[0].set_xlabel("angle v [deg]"); axs[0].set_ylabel("offset u")
axs[0].set_title("(d) Radon transform R(u,v) of the GELCT plane (Eq.12)")
axs[0].legend(loc="upper right", fontsize=9)
plt.colorbar(im, ax=axs[0])

axs[1].plot(np.degrees(v_axis), R.max(axis=0), lw=1.2)
axs[1].axvline(np.degrees(rad["v_hat"]), color="crimson", ls="--",
               label=f"argmax  v={np.degrees(rad['v_hat']):.3f} deg")
axs[1].set_xlabel("angle v [deg]"); axs[1].set_ylabel(r"$\max_u R(u,v)$")
axs[1].set_title(r"(e) $k=-\cot(v)\,\Delta f/\Delta t$  =  "
                 f"{k_hat:.1f} Hz/s  (true {p.chirp_rate:.0f})")
axs[1].legend(fontsize=9); axs[1].grid(alpha=0.3)

fig2.tight_layout()
fig2.savefig(os.path.join(OUT, "demo_single_radon.png"), dpi=130)
print("saved:", os.path.join(OUT, "demo_single_tf.png"))
print("saved:", os.path.join(OUT, "demo_single_radon.png"))

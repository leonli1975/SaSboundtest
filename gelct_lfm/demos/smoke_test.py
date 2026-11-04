# -*- coding: utf-8 -*-
"""
Self-check of every building block of the GELCT estimator.
Run:  python smoke_test.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sas_noise import (sas_rvs, complex_sas_noise, kappa_alpha,
                       crlb_chirp_rate, crlb_initial_frequency)
from gelct_core import (LFMParams, lfm_signal, received_signal, nonlinear_transform,
                        gelct, GELCTConfig, radon_sinogram, preprocess_tf,
                        estimate_lfm_params, estimate_initial_frequency,
                        estimate_k_full_window, default_k_max)

N, FS, TS = 1024, 10240.0, 0.1
P = LFMParams(1.0, 1000.0, 10000.0)
CFG = GELCTConfig(win_len=256, hop=8, nfft=1024, L=32)
rng = np.random.default_rng(20211102)

print("=" * 72)
print("1) S-alpha-S generator (Chambers-Mallows-Stuck)")
x2 = sas_rvs(2.0, 1.0, 2_000_000, rng)
print(f"   alpha=2  : sample variance = {x2.var():.4f}   (theory 2*gamma = 2)")
x1 = sas_rvs(1.0, 1.0, 2_000_000, rng)
print(f"   alpha=1  : median|X| = {np.median(np.abs(x1)):.4f}  (Cauchy scale 1)")
x15 = sas_rvs(1.5, 1.0, 2_000_000, rng)
print(f"   alpha=1.5: P(|X|>10) = {np.mean(np.abs(x15) > 10):.5f}  (impulsiveness)")

print("-" * 72)
print("2) kappa(alpha) = integral (f')^2/f  (paper: kappa(1)=kappa(2)=0.5)")
for a in (1.0, 1.2, 1.5, 1.8, 2.0):
    print(f"   alpha={a:<4}  kappa={kappa_alpha(a):.6f}")

print("-" * 72)
print("3) nonlinear transform, Eq. (8): amplitude compression, phase kept")
probe = np.array([1e-4, 1e-2, 0.5, 1.0, 10.0, 1e3, 1e6, 1e9]) * np.exp(1j * 0.7)
out = nonlinear_transform(probe)
for a, b in zip(probe, out):
    print(f"   |r|={abs(a):<10.0e} -> |f[r]|={abs(b):8.4f}   "
          f"phase error={abs(np.angle(b) - np.angle(a)):.2e}")

print("-" * 72)
print(f"4) estimates  (true f0={P.f0}, k={P.chirp_rate}, "
      f"k_max={default_k_max(N, FS):.0f})")
print(f"   {'case':<26}{'k_radon':>11}{'f0_radon':>11}{'k_full':>11}{'f0_full':>11}")
s = lfm_signal(N, FS, P)
res = estimate_lfm_params(s, FS, TS, CFG)
kf = estimate_k_full_window(s, FS)
print(f"   {'noise free':<26}{res['k_hat']:>11.1f}{res['f0_hat']:>11.1f}"
      f"{kf:>11.1f}{estimate_initial_frequency(s, kf, FS)[0]:>11.1f}")
for alpha in (1.0, 1.5, 2.0):
    for gsnr in (0.0, -5.0, -8.0):
        r, c = received_signal(N, FS, P, alpha=alpha, gsnr_db=gsnr, rng=rng)
        res = estimate_lfm_params(r, FS, TS, CFG)
        kf = estimate_k_full_window(r, FS)
        f0f = estimate_initial_frequency(r, kf, FS)[0]
        print(f"   a={alpha} GSNR={gsnr:>5.1f}dB A={c['amplitude']:.3f}   "
              f"{res['k_hat']:>11.1f}{res['f0_hat']:>11.1f}{kf:>11.1f}{f0f:>11.1f}")

print("-" * 72)
print("5) CRLB  (Eq. 44, 45)")
for a in (1.0, 1.5, 2.0):
    kap = kappa_alpha(a)
    for gsnr_db in (0.0, -5.0):
        g = 10 ** (gsnr_db / 10)
        print(f"   alpha={a} GSNR={gsnr_db:>5.1f}dB  "
              f"std(k)>={np.sqrt(crlb_chirp_rate(N, 1.0/FS, kap, g)):.3e}  "
              f"std(f0)>={np.sqrt(crlb_initial_frequency(N, 1.0/FS, kap, g)):.3e}")
print("=" * 72)

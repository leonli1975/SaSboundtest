# -*- coding: utf-8 -*-
"""诊断：白噪声（复高斯 / SαS）下三个估计器的 CRLB 对比，确认实现正确性。"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_inject import (chirp, ls_estimate, huber_estimate, mle_estimate,
                         SasLikelihood, crlb, sigma_med, PRF, N, T_CPI, K_TRUE,
                         C)

rng = np.random.default_rng(7)
n = np.arange(N)
snrs = [-6, -2, 2, 6, 10]
print(f"CRLB(f0): " + ", ".join(f"{s}dB:{crlb(10**(s/10), n)[0]:.3f}" for s in snrs))
print(f"{'噪声':<10}{'SNR':>5}{'LS':>10}{'Huber':>10}{'MLE':>10}{'CRLB':>8}")

for kind, alpha in (("gauss", 2.0), ("sas1.6", 1.6), ("sas1.3", 1.3)):
    lik = SasLikelihood(alpha) if kind != "gauss" else None
    for snr in snrs:
        err = {"LS": [], "Huber": [], "MLE": []}
        for tr in range(40):
            if kind == "gauss":
                w = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
            else:
                u = C.sas_quantile(rng.random(2 * N), alpha)
                w = (u[::2] + 1j * u[1::2]) / np.sqrt(2)
            sig = sigma_med(w)
            A = np.sqrt(10 ** (snr / 10) * sig ** 2)
            f0 = rng.uniform(0.1, 0.3) * PRF
            k = K_TRUE
            x = w + A * chirp(n, f0, k, rng.uniform(0, 2 * np.pi))
            e1 = ls_estimate(x, n)
            e2 = huber_estimate(x, n, e1)
            e3 = mle_estimate(x, n, lik, e1) if lik else (np.nan, np.nan)
            err["LS"].append(e1[0] - f0); err["Huber"].append(e2[0] - f0); err["MLE"].append(e3[0] - f0)
        row = f"{kind:<10}{snr:>5}"
        for k in ("LS", "Huber", "MLE"):
            row += f"{np.sqrt(np.mean(np.square(err[k]))):>10.2f}"
        row += f"{crlb(10**(snr/10), n)[0]:>8.2f}"
        print(row, flush=True)

# 杂波谱形状检查：2022 HH 一个单元的功率谱
import scipy.io as sio
d = sio.loadmat(r"F:\海杂波数据\T2exp\MTDSP2022_HH_stare.mat",
                squeeze_me=True, struct_as_record=False,
                variable_names=["pc", "clutter_cells"])
cc = np.atleast_1d(d["clutter_cells"]).astype(int) - 1
v = d["pc"][:4096, cc[len(cc) // 2]]
P = np.abs(np.fft.fftshift(np.fft.fft(v * np.hanning(v.size), 8192))) ** 2
f = np.fft.fftshift(np.fft.fftfreq(8192, 1 / PRF))
Pdb = 10 * np.log10(P / P.max() + 1e-12)
print("\n杂波谱（相对峰值 dB）@ 0/±100/±500/±1000 Hz:",
      [round(float(Pdb[np.argmin(abs(f - q))]), 1) for q in (0, 100, -100, 500, -500, 1000, -1000)])

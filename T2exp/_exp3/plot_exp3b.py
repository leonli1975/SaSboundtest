# -*- coding: utf-8 -*-
"""Step3 两张干净图：合成 SαS 一张、真实杂波一张（RMSE 与 MAD 双纵轴）。"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3_inject import crlb, N

DIR = r"F:\海杂波数据\T2exp\_exp3"
rows = json.load(open(os.path.join(DIR, "exp3_rows.json"), encoding="utf-8"))


def curve(flt, est, metric):
    g = [r for r in rows if flt(r) and r["est"] == est]
    if not g:
        return None, None
    snrs = sorted({r["snr_db"] for r in g})
    v = []
    for s in snrs:
        d = np.array([r["df"] for r in g if r["snr_db"] == s])
        if metric == "rmse":
            v.append(float(np.sqrt(np.mean(d ** 2))))
        else:
            v.append(float(np.median(np.abs(d - np.median(d)))))
    return np.array(snrs), np.array(v)


snr_grid = np.array([-10, -6, -2, 2, 6, 10], float)
cr = np.array([crlb(10 ** (s / 10), np.arange(N))[0] for s in snr_grid])
STY = {"LS": ("o", "-", 2.0), "Huber": ("s", "--", 1.6), "MLE": ("^", "-.", 1.6)}

# ---------- 图 1：合成 SαS
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
for a, col in ((1.3, "tab:red"), (1.6, "tab:orange"), (1.9, "tab:green")):
    for est in ("LS", "Huber", "MLE"):
        s, v = curve(lambda r, a=a: r["file"].startswith("synth") and abs(r["alpha"] - a) < 1e-6, est, "rmse")
        if s is None:
            continue
        m, dash, lw = STY[est]
        ax[0].semilogy(s, np.maximum(v, 1e-3), marker=m, ls=dash, lw=lw, color=col,
                       ms=4, alpha=.9, label=f"α={a} {est}")
ax[0].semilogy(snr_grid, cr, "k:", lw=2, label="Gaussian CRLB")
ax[0].set_xlabel("per-sample SNR / dB"); ax[0].set_ylabel("RMSE of $f_0$ / Hz")
ax[0].set_title("synthetic SαS: RMSE"); ax[0].grid(alpha=.3, which="both")
ax[0].legend(fontsize=6.4, ncol=2)

for a, col in ((1.3, "tab:red"), (1.6, "tab:orange"), (1.9, "tab:green")):
    for est in ("LS", "Huber", "MLE"):
        s, v = curve(lambda r, a=a: r["file"].startswith("synth") and abs(r["alpha"] - a) < 1e-6, est, "mad")
        if s is None:
            continue
        m, dash, lw = STY[est]
        ax[1].semilogy(s, np.maximum(v, 1e-3), marker=m, ls=dash, lw=lw, color=col, ms=4, alpha=.9)
ax[1].semilogy(snr_grid, cr, "k:", lw=2)
ax[1].set_xlabel("per-sample SNR / dB"); ax[1].set_ylabel("MAD of $f_0$ / Hz")
ax[1].set_title("synthetic SαS: MAD"); ax[1].grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(DIR, "exp3_synth.png"), dpi=150)
print("-> exp3_synth.png")

# ---------- 图 2：真实杂波
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
for f, col in (("MTDSP2022_HH_stare.mat", "tab:blue"), ("MTDSP2022_VV_stare.mat", "tab:brown")):
    for est in ("LS", "Huber", "MLE"):
        s, v = curve(lambda r, f=f: r["file"] == f, est, "rmse")
        m, dash, lw = STY[est]
        ax[0].semilogy(s, np.maximum(v, 1e-3), marker=m, ls=dash, lw=lw, color=col, ms=4,
                       alpha={"LS": 1.0, "Huber": .65, "MLE": .45}[est],
                       label=f"{f.split('_')[1]} {est}")
ax[0].semilogy(snr_grid, cr, "k:", lw=2, label="Gaussian CRLB")
ax[0].set_xlabel("per-sample SNR / dB"); ax[0].set_ylabel("RMSE of $f_0$ / Hz")
ax[0].set_title("real sea clutter (2022 stare)"); ax[0].grid(alpha=.3, which="both")
ax[0].legend(fontsize=6.6, ncol=2)

for f, col in (("MTDSP2022_HH_stare.mat", "tab:blue"), ("MTDSP2022_VV_stare.mat", "tab:brown")):
    for est in ("LS", "Huber", "MLE"):
        s, v = curve(lambda r, f=f: r["file"] == f, est, "mad")
        m, dash, lw = STY[est]
        ax[1].semilogy(s, np.maximum(v, 1e-3), marker=m, ls=dash, lw=lw, color=col, ms=4,
                       alpha={"LS": 1.0, "Huber": .65, "MLE": .45}[est])
ax[1].semilogy(snr_grid, cr, "k:", lw=2)
ax[1].set_xlabel("per-sample SNR / dB"); ax[1].set_ylabel("MAD of $f_0$ / Hz")
ax[1].set_title("real sea clutter: MAD"); ax[1].grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(DIR, "exp3_real.png"), dpi=150)
print("-> exp3_real.png")

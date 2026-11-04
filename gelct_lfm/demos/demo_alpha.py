# -*- coding: utf-8 -*-
"""
NRMSE of the GELCT estimator versus the characteristic exponent alpha, at a
fixed GSNR  --  the spirit of Fig. 3 of the paper (robustness to impulsiveness).

Usage:
    python demo_alpha.py --trials 200 --gsnr 0 -5 --alphas 0.4 2.0 0.2
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from sas_noise import kappa_alpha, crlb_chirp_rate, crlb_initial_frequency
from gelct_core import (LFMParams, received_signal, gelct, GELCTConfig,
                        estimate_chirp_rate, estimate_initial_frequency,
                        estimate_k_full_window)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
N, FS, TS = 1024, 10240.0, 0.1
P = LFMParams(1.0, 1000.0, 10000.0)
CFG = GELCTConfig(win_len=256, hop=8, nfft=1024, L=32)


def job(args):
    alpha, gsnr, seed = args
    r, _ = received_signal(N, FS, P, alpha=alpha, gsnr_db=gsnr,
                           rng=np.random.default_rng(seed))
    tf = gelct(r, FS, TS, CFG)
    k_r, _ = estimate_chirp_rate(tf)
    k_f = estimate_k_full_window(r, FS)
    return k_r, k_f


def nrmse_db(est, true_val):
    e = np.asarray(est, float)
    return 10.0 * np.log10(np.mean((e - true_val) ** 2) / true_val ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--gsnr", type=float, nargs="+", default=[0.0, -5.0])
    ap.add_argument("--alphas", type=float, nargs=3, default=[0.4, 2.0, 0.2])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="alpha")
    a = ap.parse_args()

    alphas = np.round(np.arange(a.alphas[0], a.alphas[1] + 1e-9, a.alphas[2]), 3)
    print(f"trials={a.trials}  alphas={list(alphas)}  GSNR={a.gsnr}", flush=True)

    fig, axes = plt.subplots(1, len(a.gsnr), figsize=(6.2 * len(a.gsnr), 4.8),
                             squeeze=False)
    res = {}
    for ax, gsnr in zip(axes[0], a.gsnr):
        jobs = [(al, gsnr, 900000 + int(al * 1000) * 137 + s)
                for al in alphas for s in range(a.trials)]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            out = list(ex.map(job, jobs, chunksize=4))
        krad, kfull, ck, cf = [], [], [], []
        for i, al in enumerate(alphas):
            ch = out[i * a.trials:(i + 1) * a.trials]
            krad.append(nrmse_db([c[0] for c in ch], P.chirp_rate))
            kfull.append(nrmse_db([c[1] for c in ch], P.chirp_rate))
            kap = kappa_alpha(float(al))
            gl = 10 ** (gsnr / 10.0)
            ck.append(10 * np.log10(crlb_chirp_rate(N, 1.0/FS, kap, gl) / P.chirp_rate ** 2))
        res[gsnr] = (alphas, krad, kfull, ck)
        print(f"  GSNR={gsnr} done in {time.time()-t0:.1f} s", flush=True)

        ax.plot(alphas, krad, "o-", label="GELCT + Radon (short-time)")
        ax.plot(alphas, kfull, "s--", mfc="none", label="GELCT full-window")
        ax.plot(alphas, ck, ":", lw=1.6, label="CRLB(k)")
        ax.axhline(-20, color="grey", lw=0.8, ls="--")
        ax.set_xlabel(r"characteristic exponent $\alpha$")
        ax.set_ylabel("NRMSE(k) [dB]  (Eq. 52)")
        ax.set_title(f"GSNR = {gsnr} dB")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Robustness of the GELCT estimator to impulsiveness "
                 f"({a.trials} runs/point)", fontsize=13)
    fig.tight_layout()
    png = os.path.join(OUT, f"{a.tag}.png")
    fig.savefig(png, dpi=130)
    np.save(os.path.join(OUT, f"{a.tag}_data.npy"), res, allow_pickle=True)
    print("saved:", png, flush=True)


if __name__ == "__main__":
    main()

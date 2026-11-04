# -*- coding: utf-8 -*-
"""
Monte-Carlo NRMSE of the GELCT estimator versus GSNR, compared with the CRLBs
(Eq. (44)-(51)).  Reproduces the spirit of Figs. 1, 2 and 5 of the paper.

Two estimator variants are evaluated:

  *  "GELCT+Radon"     - Algorithm 1 as printed: short-time GELCT plane (Eq. 7)
                         followed by the Radon transform (Eq. 12-14).
  *  "GELCT full-window" - the same transform with win_len = N, for which the
                         Radon stage collapses onto a coherent 1-D de-chirp scan
                         (no window loss in processing gain).

NRMSE is the paper's Eq. (52), reported in dB:
    NRMSE = 10 lg[ sum_i (Y - Yhat_i)^2 / (M * Y^2) ]

Usage:
    python demo_nrmse.py --trials 200 --alphas 1 1.5 2 --gsnr -12 8 2
"""
import sys, os, time, argparse, itertools
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
P_TRUE = LFMParams(amplitude=1.0, f0=1000.0, chirp_rate=10000.0)
CFG = GELCTConfig(win_len=256, hop=8, nfft=1024, L=32)


def one_trial(job):
    alpha, gsnr, seed = job
    rng = np.random.default_rng(seed)
    r, _ = received_signal(N, FS, P_TRUE, alpha=alpha, gsnr_db=gsnr, rng=rng)
    out = {}
    tf = gelct(r, FS, TS, CFG)
    k_r, _ = estimate_chirp_rate(tf)
    f0_r, _ = estimate_initial_frequency(r, k_r, FS)
    out["radon"] = (k_r, f0_r)
    k_f = estimate_k_full_window(r, FS)
    f0_f, _ = estimate_initial_frequency(r, k_f, FS)
    out["fullwin"] = (k_f, f0_f)
    return out


def nrmse_db(est, true_val):
    est = np.asarray(est, dtype=float)
    return 10.0 * np.log10(np.mean((est - true_val) ** 2) / true_val ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--gsnr", type=float, nargs=3, default=[-12.0, 8.0, 2.0],
                    metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=100000)
    ap.add_argument("--tag", default="nrmse")
    args = ap.parse_args()

    gsnrs = np.arange(args.gsnr[0], args.gsnr[1] + 1e-9, args.gsnr[2])
    print(f"trials={args.trials}  alphas={args.alphas}  GSNR={list(gsnrs)}  "
          f"workers={args.workers}", flush=True)

    store = {}
    t_all = time.time()
    for alpha in args.alphas:
        kap = kappa_alpha(alpha)
        jobs = [(alpha, g, args.seed0 + 100000 * int(alpha * 10) + s)
                for g in gsnrs for s in range(args.trials)]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(one_trial, jobs, chunksize=4))
        print(f"  alpha={alpha}: {len(jobs)} runs in {time.time()-t0:.1f} s", flush=True)
        rec = {}
        for i, g in enumerate(gsnrs):
            chunk = results[i * args.trials:(i + 1) * args.trials]
            k_rad = [c["radon"][0] for c in chunk]
            f_rad = [c["radon"][1] for c in chunk]
            k_fw = [c["fullwin"][0] for c in chunk]
            f_fw = [c["fullwin"][1] for c in chunk]
            gl = 10 ** (g / 10.0)
            rel = lambda e, t: float(np.median(np.abs(np.array(e) - t) / t))
            hit = lambda e, t, thr=0.01: float(np.mean(np.abs(np.array(e) - t) / t < thr))
            rec[g] = {
                "k_radon": nrmse_db(k_rad, P_TRUE.chirp_rate),
                "f_radon": nrmse_db(f_rad, P_TRUE.f0),
                "k_full": nrmse_db(k_fw, P_TRUE.chirp_rate),
                "f_full": nrmse_db(f_fw, P_TRUE.f0),
                "k_radon_med": rel(k_rad, P_TRUE.chirp_rate),
                "f_radon_med": rel(f_rad, P_TRUE.f0),
                "k_full_med": rel(k_fw, P_TRUE.chirp_rate),
                "f_full_med": rel(f_fw, P_TRUE.f0),
                "k_radon_hit": hit(k_rad, P_TRUE.chirp_rate, 0.03),
                "k_full_hit": hit(k_fw, P_TRUE.chirp_rate, 0.03),
                "f_radon_hit": hit(f_rad, P_TRUE.f0, 0.03),
                "f_full_hit": hit(f_fw, P_TRUE.f0, 0.03),
                "crlb_k": 10 * np.log10(crlb_chirp_rate(N, 1.0/FS, kap, gl) / P_TRUE.chirp_rate ** 2),
                "crlb_f": 10 * np.log10(crlb_initial_frequency(N, 1.0/FS, kap, gl) / P_TRUE.f0 ** 2),
            }
        store[alpha] = rec
        for g in gsnrs:
            d = rec[g]
            print(f"   a={alpha} GSNR={g:>6.1f} | k: radon {d['k_radon']:8.1f}dB "
                  f"(med {d['k_radon_med']:.1e}, hit3 {d['k_radon_hit']*100:3.0f}%) "
                  f"full {d['k_full']:8.1f}dB (med {d['k_full_med']:.1e}, "
                  f"hit3 {d['k_full_hit']*100:3.0f}%) crlb {d['crlb_k']:8.1f} | "
                  f"f0: radon {d['f_radon']:8.1f} (med {d['f_radon_med']:.1e}) "
                  f"full {d['f_full']:8.1f} crlb {d['crlb_f']:8.1f}", flush=True)

    np.save(os.path.join(OUT, f"{args.tag}_data.npy"),
            {"gsnrs": gsnrs, "store": store}, allow_pickle=True)

    # ---------------- figure ------------------------------------------------
    n = len(args.alphas)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    cols = plt.cm.viridis(np.linspace(0.1, 0.8, n))
    for c, alpha in zip(cols, args.alphas):
        rec = store[alpha]
        g = np.array(sorted(rec))
        axes[0].plot(g, [rec[x]["k_radon"] for x in g], "o-", color=c,
                     label=f"GELCT+Radon, $\\alpha$={alpha}")
        axes[0].plot(g, [rec[x]["k_full"] for x in g], "s--", color=c, mfc="none",
                     label=f"GELCT full-window, $\\alpha$={alpha}")
        axes[0].plot(g, [rec[x]["crlb_k"] for x in g], ":", color=c, lw=1.6,
                     label=f"CRLB(k), $\\alpha$={alpha}")
        axes[1].plot(g, [rec[x]["f_radon"] for x in g], "o-", color=c,
                     label=f"GELCT+Radon, $\\alpha$={alpha}")
        axes[1].plot(g, [rec[x]["f_full"] for x in g], "s--", color=c, mfc="none",
                     label=f"GELCT full-window, $\\alpha$={alpha}")
        axes[1].plot(g, [rec[x]["crlb_f"] for x in g], ":", color=c, lw=1.6,
                     label=f"CRLB($f_0$), $\\alpha$={alpha}")
    for a, ttl in zip(axes, ["chirp rate  $k$", "initial frequency  $f_0$"]):
        a.set_xlabel("GSNR [dB]"); a.set_ylabel("NRMSE [dB]  (Eq. 52)")
        a.set_title(ttl); a.grid(alpha=0.3); a.legend(fontsize=7, ncol=2)
    fig.suptitle(f"LFM parameter estimation in S$\\alpha$S noise "
                 f"({args.trials} Monte-Carlo runs per point)", fontsize=13)
    fig.tight_layout()
    png = os.path.join(OUT, f"{args.tag}.png")
    fig.savefig(png, dpi=130)
    print("saved:", png, " total", round(time.time() - t_all, 1), "s", flush=True)


if __name__ == "__main__":
    main()

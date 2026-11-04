# -*- coding: utf-8 -*-
"""
Controlled test of the claim

    "In SalphaS clutter, GELCT collapses exactly like least squares (LS)."

Design -- a single-variable manipulation
----------------------------------------
For the complex LFM model of Eq. (1)-(2) the *approximate least squares* (i.e.
periodogram / de-chirp) estimator of (k, f0) is

    (k_hat, f0_hat) = argmax_{k,f0} | sum_n r[n] exp(-j pi k n^2/fs^2)
                                            exp(-j 2 pi f0 n/fs) |

(a)  LS   : applied to the raw received signal r[n].
(b)  GELCT: applied to f[r[n]] of Eq. (8) -- identical code path, identical
            search grid, identical data.  The ONLY difference is the pointwise
            nonlinear transform.

So the comparison isolates exactly the mechanism that is supposed to prevent
the collapse.  If "GELCT collapses like LS" were true, (a) and (b) would have to
behave the same.

Metrics per (alpha, GSNR) cell, over `--trials` Monte-Carlo runs:
  * NRMSE      = 10 lg[ mean (Y-Yhat)^2 / Y^2 ]      (paper Eq. 52)
  * median     = median relative error               (robust location)
  * grossfail  = P(relative error > 10%)             ("collapse" rate)
  * divergence = P(relative error > 100%)

Usage
    python collapse_test.py --trials 150 --alphas 0.8 1.0 1.2 1.5 2.0 \
                            --gsnr -20 10 5 --workers 13
"""
from __future__ import annotations

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from sas_noise import sas_rvs, complex_sas_noise, kappa_alpha, crlb_chirp_rate
from gelct_core import LFMParams, received_signal, nonlinear_transform, _parabolic

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

N, FS, TS = 1024, 10240.0, 0.1
P_TRUE = LFMParams(1.0, 1000.0, 10000.0)
K_MAX = FS ** 2 / (2.0 * N)          # 51200 Hz/s : sweeps the full band


# ---------------------------------------------------------------------------
# the two estimators -- identical apart from the nonlinear transform
# ---------------------------------------------------------------------------
def _dechirp_peak(z0: np.ndarray, fs: float, kg: np.ndarray, chunk: int = 512):
    """max_w |FFT( z0(t) exp(-j pi k t^2) )|  for every k in kg."""
    t2 = (np.arange(z0.size) / fs) ** 2
    sc = np.empty(kg.size)
    for a in range(0, kg.size, chunk):
        sub = kg[a:a + chunk]
        Z = z0[:, None] * np.exp(-1j * np.pi * sub[None, :] * t2[:, None])
        sc[a:a + chunk] = np.abs(np.fft.fft(Z, axis=0)).max(axis=0)
    return sc


def _search(z0, fs, n_coarse=1601, n_mid=401, n_fine=201):
    kg = np.linspace(-K_MAX, K_MAX, n_coarse)
    c1 = _dechirp_peak(z0, fs, kg)
    k1 = _parabolic(kg, c1, int(np.argmax(c1)))
    span = 2.0 * (kg[1] - kg[0])
    kg2 = np.linspace(k1 - span, k1 + span, n_mid)
    c2 = _dechirp_peak(z0, fs, kg2)
    k2 = _parabolic(kg2, c2, int(np.argmax(c2)))
    kg3 = np.linspace(k2 - 0.25 * span, k2 + 0.25 * span, n_fine)
    c3 = _dechirp_peak(z0, fs, kg3)
    return _parabolic(kg3, c3, int(np.argmax(c3)))


def estimate_ls(r: np.ndarray, fs: float):
    """(a) approximate least squares / periodogram on the raw signal."""
    return _search(np.asarray(r, dtype=complex), fs)


def estimate_gelct(r: np.ndarray, fs: float):
    """(b) the same search on the Eq. (8) transformed signal."""
    return _search(nonlinear_transform(r), fs)


def initial_frequency(r: np.ndarray, k_hat: float, fs: float, transformed: bool):
    z0 = nonlinear_transform(r) if transformed else np.asarray(r, dtype=complex)
    t = np.arange(r.size) / fs
    S = np.abs(np.fft.fft(z0 * np.exp(-1j * np.pi * k_hat * t ** 2)))
    fr = np.fft.fftfreq(r.size, d=1.0 / fs)
    j = int(np.argmax(S))
    return float(fr[j])


# ---------------------------------------------------------------------------
# one Monte-Carlo cell
# ---------------------------------------------------------------------------
def _trial(job):
    alpha, gsnr, seed, quad = job
    r, _ = received_signal(N, FS, P_TRUE, alpha=alpha, gsnr_db=gsnr,
                           rng=np.random.default_rng(seed), quadrature=quad)
    k_ls = estimate_ls(r, FS)
    f_ls = initial_frequency(r, k_ls, FS, transformed=False)
    k_ge = estimate_gelct(r, FS)
    f_ge = initial_frequency(r, k_ge, FS, transformed=True)
    return (k_ls, f_ls, k_ge, f_ge,
            float(np.max(np.abs(r))),                       # largest sample
            float(np.max(np.abs(nonlinear_transform(r)))))  # after Eq. (8)


def _stats(est, true_v):
    e = np.asarray(est, float)
    rel = np.abs(e - true_v) / abs(true_v)
    return {
        "nrmse": float(10.0 * np.log10(max(np.mean((e - true_v) ** 2)
                                           / true_v ** 2, 1e-300))),
        "median": float(np.median(rel)),
        "mean": float(np.mean(rel)),
        "grossfail": float(np.mean(rel > 0.10)),
        "divergence": float(np.mean(rel > 1.00)),
        "huge": float(np.mean(rel > 10.0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.8, 1.0, 1.2, 1.5, 2.0])
    ap.add_argument("--gsnr", type=float, nargs=3, default=[-20.0, 10.0, 5.0],
                    metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("--quadrature", default="real",
                    choices=["real", "complex"])
    ap.add_argument("--workers", type=int, default=13)
    ap.add_argument("--seed0", type=int, default=770000)
    ap.add_argument("--tag", default="collapse")
    a = ap.parse_args()

    gsnrs = np.arange(a.gsnr[0], a.gsnr[1] + 1e-9, a.gsnr[2])
    print(f"trials={a.trials} alphas={a.alphas} GSNR={list(gsnrs)} "
          f"quadrature={a.quadrature} workers={a.workers}", flush=True)

    store = {}
    t_all = time.time()
    for alpha in a.alphas:
        jobs = [(alpha, g,
                 a.seed0 + int(round(alpha * 100)) * 1_000_000
                 + int(round((g + 100) * 10)) * 1_000 + s, a.quadrature)
                for g in gsnrs for s in range(a.trials)]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(_trial, jobs, chunksize=4))
        print(f"  alpha={alpha}: {len(jobs)} runs in {time.time()-t0:.1f} s", flush=True)
        rec = {}
        for i, g in enumerate(gsnrs):
            ch = res[i * a.trials:(i + 1) * a.trials]
            ls_k = [c[0] for c in ch]
            ge_k = [c[2] for c in ch]
            rec[g] = {
                "ls_k": _stats(ls_k, P_TRUE.chirp_rate),
                "ge_k": _stats(ge_k, P_TRUE.chirp_rate),
                "ls_f": _stats([c[1] for c in ch], P_TRUE.f0),
                "ge_f": _stats([c[3] for c in ch], P_TRUE.f0),
                "max_abs": float(np.median([c[4] for c in ch])),
                "max_abs_tx": float(np.median([c[5] for c in ch])),
                "crlb_k": 10 * np.log10(crlb_chirp_rate(N, TS, kappa_alpha(alpha), 10 ** (g / 10))
                                        / P_TRUE.chirp_rate ** 2),
            }
        store[alpha] = rec
        for g in gsnrs:
            d = rec[g]
            print(f"   a={alpha} GSNR={g:>6.1f} | LS  k: {d['ls_k']['nrmse']:9.2f}dB "
                  f"med {d['ls_k']['median']:8.2e} fail {d['ls_k']['grossfail']*100:5.1f}% "
                  f"div {d['ls_k']['divergence']*100:5.1f}% | GELCT k: "
                  f"{d['ge_k']['nrmse']:9.2f}dB med {d['ge_k']['median']:8.2e} "
                  f"fail {d['ge_k']['grossfail']*100:5.1f}% div {d['ge_k']['divergence']*100:5.1f}% "
                  f"| max|r| {d['max_abs']:.2e} -> max|f[r]| {d['max_abs_tx']:.1f}", flush=True)

    np.save(os.path.join(OUT, f"{a.tag}_data.npy"),
            {"gsnrs": gsnrs, "store": store, "quadrature": a.quadrature},
            allow_pickle=True)

    # ---------------- figure ------------------------------------------------
    n = len(a.alphas)
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, n))
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.9))
    for c, al in zip(cols, a.alphas):
        rec = store[al]
        g = np.array(sorted(rec))
        ax[0].plot(g, [rec[x]["ls_k"]["nrmse"] for x in g], "o-", color=c,
                   label=f"LS, $\\alpha$={al}")
        ax[0].plot(g, [rec[x]["ge_k"]["nrmse"] for x in g], "s--", mfc="none",
                   color=c, label=f"GELCT, $\\alpha$={al}")
        ax[1].plot(g, [100 * rec[x]["ls_k"]["grossfail"] for x in g], "o-", color=c,
                   label=f"LS, $\\alpha$={al}")
        ax[1].plot(g, [100 * rec[x]["ge_k"]["grossfail"] for x in g], "s--",
                   mfc="none", color=c, label=f"GELCT, $\\alpha$={al}")
        ax[2].plot(g, [rec[x]["max_abs"] for x in g], "o-", color=c,
                   label=f"$\\max|r|$, $\\alpha$={al}")
        ax[2].plot(g, [rec[x]["max_abs_tx"] for x in g], "s--", mfc="none", color=c,
                   label=f"$\\max|f[r]|$, $\\alpha$={al}")
    ax[0].set_xlabel("GSNR [dB]"); ax[0].set_ylabel("NRMSE(k) [dB], Eq. (52)")
    ax[0].set_title("(a) estimation error  --  LS vs GELCT")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7, ncol=2)
    ax[1].set_xlabel("GSNR [dB]"); ax[1].set_ylabel("% of runs with rel.err > 10%")
    ax[1].set_title("(b) gross-failure ('collapse') rate")
    ax[1].set_ylim(-3, 103); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7, ncol=2)
    ax[2].set_yscale("log"); ax[2].set_xlabel("GSNR [dB]")
    ax[2].set_ylabel("median of the run's extreme sample")
    ax[2].set_title("(c) what Eq. (8) does to the largest sample")
    ax[2].grid(alpha=0.3, which="both"); ax[2].legend(fontsize=7, ncol=2)
    fig.suptitle("Is GELCT collapsing like LS in S$\\alpha$S clutter?  "
                 f"({a.trials} runs/point, quadrature={a.quadrature})", fontsize=13)
    fig.tight_layout()
    png = os.path.join(OUT, f"{a.tag}.png")
    fig.savefig(png, dpi=130)
    print("saved:", png, " total", round(time.time() - t_all, 1), "s", flush=True)


if __name__ == "__main__":
    main()

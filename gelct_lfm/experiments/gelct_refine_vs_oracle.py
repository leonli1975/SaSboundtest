# -*- coding: utf-8 -*-
"""Does a blind GELCT + LS/Huber refine reach the CRLB?

Estimators (all on the same received signal):
  g   : GELCT full-window alone (blind, coarse)
  ls  : GELCT coarse -> LS (coarse FFT + Nelder-Mead refine), blind
  hu  : GELCT coarse -> Huber (IRLS refine), blind
  ols : LS initialized at the true (f0,k) (oracle)
  ohu : Huber initialized at the true (f0,k) (oracle)

Metric: median absolute error / CRLB (normalized) and the gross failure rate.
"""
import os, sys, json, argparse
import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "src"))
sys.path.append(os.path.join(HERE, "..", "..", "sim"))
from gelct_core import (LFMParams, received_signal, estimate_k_full_window,
                        estimate_initial_frequency)
from crlb_lfm import gram_single, crlb, Jmap

N, FS, TS = 1024, 10240.0, 0.1
T_CPI = N / FS
FT, KT = 1000.0, 10000.0
FN, KN = FT / FS, KT / FS ** 2
NF = 2048
P = LFMParams(1.0, FT, KT)
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)
KS = ["g", "ls", "hu", "ols", "ohu"]


def chirp(n, f0, k, phi=0.0):
    t = n / FS
    return np.exp(1j * (2 * np.pi * f0 * t + np.pi * k * t * t + phi))


def to_norm(f0, k):
    return [f0 / FS, k * T_CPI / FS]


def to_phys(q):
    return q[0] * FS, q[1] * FS / T_CPI


def prof_ls(x, f0, k, n):
    s = chirp(n, f0, k)
    c = np.vdot(s, x)
    return abs(c) ** 2, c / max(n.size, 1)


def ls_refine(x, n, f0, k):
    obj = lambda q: -prof_ls(x, *to_phys(q), n)[0]
    r = minimize(obj, to_norm(f0, k), method="Nelder-Mead",
                 options=dict(xatol=1e-10, fatol=1e-12, maxiter=900))
    return to_phys(r.x)


def ls_estimate(x, n, k0):
    ks = k0 + np.linspace(-(0.3 * abs(k0) + 200), 0.3 * abs(k0) + 200, 41)
    best = (-1.0, 0.0, 0.0)
    w = np.hanning(n.size)
    for k in ks:
        y = x * np.exp(-1j * np.pi * k * (n / FS) ** 2) * w
        F = np.abs(np.fft.fft(y, NF)) ** 2
        i = int(F.argmax())
        f0 = (i if i < NF // 2 else i - NF) * FS / NF
        if F[i] > best[0]:
            best = (F[i], f0, k)
    _, f0, k = best
    return ls_refine(x, n, f0, k)


def huber_estimate(x, n, init, c=1.345, iters=8):
    p0 = to_phys(to_norm(*init))
    s0 = chirp(n, p0[0], p0[1])
    a0 = np.mean(s0.conj() * x)
    sc = 1.4826 * np.median(np.abs(x - a0 * s0)) + 1e-12

    def loss(q):
        p = to_phys(q)
        s = chirp(n, p[0], p[1])
        w = np.ones(n.size)
        for _ in range(iters):
            a = np.sum(w * x * s.conj()) / max(np.sum(w), 1e-12)
            u = np.abs(x - a * s) / sc
            w = np.where(u <= c, 1.0, c / np.maximum(u, 1e-12))
        a = np.sum(w * x * s.conj()) / max(np.sum(w), 1e-12)
        u = np.abs(x - a * s) / sc
        return float(np.sum(np.where(u <= c, 0.5 * u ** 2, c * (u - 0.5 * c))))

    r = minimize(loss, to_norm(*init), method="Nelder-Mead",
                 options=dict(xatol=1e-9, fatol=1e-10, maxiter=600))
    return to_phys(r.x)


def err(f_phys, k_phys):
    return abs(f_phys / FS - FN), abs(k_phys / FS ** 2 - KN)


def run_one(a):
    alpha, snr, seed = a
    rng = np.random.default_rng(seed)
    r, _ = received_signal(N, FS, P, alpha=alpha, gsnr_db=snr, rng=rng)
    k0 = estimate_k_full_window(r, FS)
    f0, _ = estimate_initial_frequency(r, k0, FS)
    n = np.arange(N)
    fl, kl = ls_estimate(r, n, k0)
    fh, kh = huber_estimate(r, n, (f0, k0))
    fol, kol = ls_estimate(r, n, KT)
    foh, koh = huber_estimate(r, n, (FT, KT))
    return dict(alpha=alpha, snr=snr, g=err(f0, k0), ls=err(fl, kl),
                hu=err(fh, kh), ols=err(fol, kol), ohu=err(foh, koh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 1.5])
    ap.add_argument("--gsnr", type=float, nargs="+", default=[-10.0, 0.0, 10.0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=660000)
    args = ap.parse_args()

    rows = []
    for ai, alpha in enumerate(args.alphas):
        G = gram_single(N, FN, KN)
        C, _ = crlb(G, Jmap[alpha] if alpha in Jmap else Jmap[1.10], [0, 1])
        cf, ck = float(np.sqrt(C[0, 0])), float(np.sqrt(C[1, 1]))
        for si, snr in enumerate(args.gsnr):
            jobs = [(alpha, snr, args.seed0 + 100000 * ai + 1000 * si + t)
                    for t in range(args.trials)]
            with Pool(args.workers) as pool:
                out = pool.map(run_one, jobs)
            rows.append(dict(alpha=alpha, snr=snr, cf=cf, ck=ck,
                             trials=args.trials, data=out))
            print(f"alpha={alpha} snr={snr:+.0f} done", flush=True)

    json.dump(rows, open(os.path.join(RES, "gelct_refine.json"), "w"), indent=1)

    print("\n=== summary (median MAD/CRLB) ===")
    for row in rows:
        cf, ck = row["cf"], row["ck"]
        print(f"alpha={row['alpha']} snr={row['snr']:+.0f}")
        for k in KS:
            ef = np.median([x[k][0] for x in row["data"]]) / cf
            ek = np.median([x[k][1] for x in row["data"]]) / ck
            fail = np.mean([(x[k][1] / KN) > 0.1 for x in row["data"]]) * 100
            print(f"  {k:4s} MAD_f/CRLB={ef:8.2f}  MAD_k/CRLB={ek:8.2f}  fail_k={fail:5.1f}%")
    print("\nwrote results/gelct_refine.json")


if __name__ == "__main__":
    main()

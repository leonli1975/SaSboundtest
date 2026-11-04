"""Unified sub-Rayleigh alpha sweep (df=1/N): 100 trials + bootstrap CI."""
import json, os, sys, numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(__file__)
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "..", "paper1", "code"))
from multicomponent_efficiency import (ls_single, ls_two, huber_single, huber_two,
                                       sig, F1, K1, SCALE)
from crlb_lfm import gram_single, gram_multi, crlb, Jmap
from m1_quant_corrected import csas

N = 256
DF = 1.0
TRIALS = 100
ALPHAS = [1.3, 1.5, 1.7, 2.0]
RES = os.path.join(HERE, "..", "results")

def run_one(args):
    a, seed, cf1, cf2 = args
    rng = np.random.default_rng(seed)
    f2 = F1 + DF/N
    x = sig(N, F1, K1) + sig(N, f2, K1) + SCALE*csas(N, a, rng)
    fs, ks = ls_single(x, N, F1, K1)
    ft = ls_two(x, N, F1, K1, f2, K1)
    fhs, khs = huber_single(x, N, F1, K1)
    fht, kht, _, _ = huber_two(x, N, F1, K1, f2, K1)
    return dict(alpha=a, s_ls=abs(fs-F1)/cf1, t_ls=abs(ft[0]-F1)/cf2,
                s_hu=abs(fhs-F1)/cf1, t_hu=abs(fht-F1)/cf2)

def mad_ci(vals, B=1000, seed=0):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    med = float(np.median(v))
    boots = np.array([np.median(rng.choice(v, size=v.size, replace=True))
                      for _ in range(B)])
    return med, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

if __name__ == "__main__":
    rows = []
    for a in ALPHAS:
        G1 = gram_single(N, F1, K1)
        C1, _ = crlb(G1, Jmap[a], [0, 1])
        Gm = gram_multi(N, [(F1, K1, 1.0, 0.0), (F1 + DF/N, K1, 1.0, 0.0)])
        C2, _ = crlb(Gm, Jmap[a], [0, 1])
        cf1, cf2 = float(np.sqrt(C1[0, 0])), float(np.sqrt(C2[0, 0]))
        args = [(a, 300000 + i, cf1, cf2) for i in range(TRIALS)]
        with Pool(4) as pool:
            out = pool.map(run_one, args)
        row = dict(alpha=a, cf1=cf1, cf2=cf2, trials=TRIALS,
                   var_ratio=cf2**2/cf1**2, std_ratio=cf2/cf1)
        for key, lab in [("s_ls", "single LS"), ("t_ls", "two LS"),
                         ("s_hu", "single Huber"), ("t_hu", "two Huber")]:
            vals = [o[key] for o in out]
            med, lo, hi = mad_ci(vals, seed=41 + int(10*a))
            row[key] = dict(mad=med, lo=lo, hi=hi)
            print(f"alpha={a:.1f} {lab:13s} MAD/CRLB={med:.3f} [{lo:.3f}, {hi:.3f}]")
        rows.append(row)
    json.dump(rows, open(os.path.join(RES, "unified_multi_alpha.json"), "w"), indent=1)
    print("\nwrote results/unified_multi_alpha.json")

"""Unified single-component run: 100 trials + bootstrap CI per estimator."""
import json, os, sys, numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize

HERE = os.path.dirname(__file__)
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "..", "paper1", "code"))
from estimators_sas_fast import ls_fit, huber_fit, sig, F0, K0, SCALE, ls_obj
from estimators_mle_mad import mle_fit, crlb_std, code_density
from estimators_frft import frft_coarse
from m1_quant_corrected import csas

N = 256
TRIALS = 100
ALPHAS = [1.3, 1.5, 1.7, 2.0]
RES = os.path.join(HERE, "..", "results")

def ls_fit_at(x, N, f0, k0):
    res = minimize(ls_obj, [f0, k0], args=(x, N), method="Nelder-Mead",
                   options={"xatol": 1e-11, "fatol": 1e-14, "maxiter": 250})
    return res.x

def run_one(args):
    a, seed, rc, logh = args
    rng = np.random.default_rng(seed)
    x = sig(N, F0, K0) + SCALE*csas(N, a, rng)
    fl, kl = ls_fit(x, N)
    fh, kh = huber_fit(x, N, fl, kl)
    fm, km = mle_fit(x, N, rc, logh)
    ff, kf = frft_coarse(x, N)
    flf, klf = ls_fit_at(x, N, ff, kf)
    return dict(alpha=a, ls=abs(fl-F0), hu=abs(fh-F0), ml=abs(fm-F0),
                fl=abs(flf-F0), fc=abs(ff-F0))

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
        rc, hc, phi, J, c, r50 = code_density(a)
        logh = np.log(np.maximum(hc, 1e-300))
        cf, ck = crlb_std(a, N)
        args = [(a, 100000 + i, rc, logh) for i in range(TRIALS)]
        with Pool(4) as pool:
            out = pool.map(run_one, args)
        row = dict(alpha=a, cf=cf, trials=TRIALS)
        for key, lab in [("ls", "LS"), ("hu", "Huber"), ("ml", "MLE"),
                         ("fl", "FrFT+LS"), ("fc", "FrFT only")]:
            vals = [o[key] for o in out]
            med, lo, hi = mad_ci(vals, seed=17 + int(10*a))
            row[key] = dict(mad=med, lo=lo, hi=hi,
                            r=med/cf, r_lo=lo/cf, r_hi=hi/cf)
            print(f"alpha={a:.1f} {lab:9s} MAD/CRLB={med/cf:.3f} "
                  f"[{lo/cf:.3f}, {hi/cf:.3f}]")
        rows.append(row)
    json.dump(rows, open(os.path.join(RES, "unified_single.json"), "w"), indent=1)
    print("\nwrote results/unified_single.json")

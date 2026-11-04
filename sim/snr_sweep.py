"""SNR sweep: MAD/CRLB, trimmed MSE, and outlier fraction for LS/Huber/MLE."""
import json, os, sys, numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(__file__)
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "..", "paper1", "code"))
from estimators_sas_fast import ls_fit, huber_fit, sig, F0, K0, ls_obj
from estimators_mle_mad import mle_fit, crlb_std, code_density
from m1_quant_corrected import csas
from scipy.optimize import minimize

N = 256
TRIALS = 100
ALPHAS = [1.3, 1.5]
SCALES = [0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8]
RES = os.path.join(HERE, "..", "results")

def run_one(args):
    a, s, seed, rc, logh = args
    rng = np.random.default_rng(seed)
    x = sig(N, F0, K0) + s*csas(N, a, rng)
    fl, kl = ls_fit(x, N)
    fh, kh = huber_fit(x, N, fl, kl)
    fm, km = mle_fit(x, N, rc, logh)
    return dict(alpha=a, scale=s, ls=abs(fl-F0), hu=abs(fh-F0), ml=abs(fm-F0))

if __name__ == "__main__":
    rows = []
    for a in ALPHAS:
        rc, hc, phi, J, c, r50 = code_density(a)
        logh = np.log(np.maximum(hc, 1e-300))
        cf, ck = crlb_std(a, N)
        for s in SCALES:
            args = [(a, s, 500000 + i, rc, logh) for i in range(TRIALS)]
            with Pool(4) as pool:
                out = pool.map(run_one, args)
            row = dict(alpha=a, scale=s, snr_db=20*np.log10(1.0/s), cf=cf)
            for key in ["ls", "hu", "ml"]:
                v = np.array([o[key] for o in out])
                r = v / cf
                med = np.median(r)
                trimmed = np.mean(np.sort(r)[:int(0.95*v.size)])
                outfrac = np.mean(r > 10.0)
                row[key] = dict(mad=float(med), trimmed_mse=float(trimmed**2),
                                outlier=float(outfrac))
            rows.append(row)
            print(f"alpha={a:.1f} snr={row['snr_db']:.1f} dB  "
                  f"LS MAD={row['ls']['mad']:.2f} (out={row['ls']['outlier']:.2f})  "
                  f"Hu={row['hu']['mad']:.2f}  MLE={row['ml']['mad']:.2f}")
    json.dump(rows, open(os.path.join(RES, "snr_sweep.json"), "w"), indent=1)
    print("wrote results/snr_sweep.json")

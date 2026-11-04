"""Dense kappa/J sweep for Fig. 1a: alpha = 1.0-2.0, 0.1 step in [1.3,1.9]."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "..", "paper1", "code"))
from kappa_multicomponent import kappa
from fisher_clip import code_density

RES = os.path.join(HERE, "..", "results")
ALPHAS = [1.0, 1.15, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

if __name__ == "__main__":
    rows = []
    for a in ALPHAS:
        k = kappa(a)
        rc, hc, phi, J, c, r50 = code_density(a)
        rows.append(dict(alpha=a, kappa=float(k), J=float(J)))
    k2 = 0.5
    J2 = next(r["J"] for r in rows if r["alpha"] == 2.0)
    for r in rows:
        r["kratio"] = k2 / r["kappa"]
        r["jratio"] = J2 / r["J"]
        r["kdb"] = 20*np.log10(r["kratio"])
        r["jdb"] = 20*np.log10(r["jratio"])
        print(f"alpha={r['alpha']:.2f} kappa={r['kappa']:.4f} "
              f"kratio={r['kratio']:.4f} ({r['kdb']:+.2f} dB) "
              f"jratio={r['jratio']:.4f} ({r['jdb']:+.2f} dB)")
    json.dump(rows, open(os.path.join(RES, "kappa_dense.json"), "w"), indent=1)
    print("wrote results/kappa_dense.json")

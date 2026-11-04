"""Multicomponent alpha sweep: single vs two-component estimators near resolution."""
import json, os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "..", "paper1", "code"))
from multicomponent_efficiency import (ls_single, ls_two, huber_single, huber_two,
                                       sig, F1, K1, SCALE)
from crlb_lfm import gram_single, gram_multi, crlb, Jmap
from m1_quant_corrected import csas

RES = os.path.join(HERE, "..", "results")

if __name__ == "__main__":
    rng = np.random.default_rng(20261108)
    N = 256
    alphas = [1.3, 1.5, 1.7, 2.0]
    dfs = [1.0, 2.0, 4.0]
    trials = 25
    rows = []
    for a in alphas:
        print(f"\n=== alpha={a:.2f}")
        for df in dfs:
            f2 = F1 + df/N
            G1 = gram_single(N, F1, K1)
            C1, _ = crlb(G1, Jmap[a], [0, 1])
            Gm = gram_multi(N, [(F1, K1, 1.0, 0.0), (f2, K1, 1.0, 0.0)])
            C2, _ = crlb(Gm, Jmap[a], [0, 1])
            e_s, e_t, e_hs, e_ht = [], [], [], []
            for t in range(trials):
                x = sig(N, F1, K1) + sig(N, f2, K1) + SCALE*csas(N, a, rng)
                fs, ks = ls_single(x, N, F1, K1)
                ft = ls_two(x, N, F1, K1, f2, K1)
                fhs, khs = huber_single(x, N, F1, K1)
                fht, kht, _, _ = huber_two(x, N, F1, K1, f2, K1)
                e_s.append(abs(fs-F1)); e_t.append(abs(ft[0]-F1))
                e_hs.append(abs(fhs-F1)); e_ht.append(abs(fht-F1))
            row = dict(alpha=a, df=df, cf1=np.sqrt(C1[0,0]), cf2=np.sqrt(C2[0,0]),
                       single_ls=np.median(e_s)/np.sqrt(C1[0,0]),
                       two_ls=np.median(e_t)/np.sqrt(C2[0,0]),
                       single_hu=np.median(e_hs)/np.sqrt(C1[0,0]),
                       two_hu=np.median(e_ht)/np.sqrt(C2[0,0]))
            rows.append(row)
            print(f"  df={df:.0f}/N  single LS={row['single_ls']:.2f} two LS={row['two_ls']:.2f}  "
                  f"single Hu={row['single_hu']:.2f} two Hu={row['two_hu']:.2f}")
    json.dump(rows, open(os.path.join(RES, "multicomponent_alpha.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for df, mk in zip(dfs, ["o","s","^"]):
        rr = [r for r in rows if r["df"] == df]
        ax[0].plot([r["alpha"] for r in rr], [r["single_ls"] for r in rr], mk+"-", label=f"single LS df={df}N")
        ax[0].plot([r["alpha"] for r in rr], [r["two_ls"] for r in rr], mk+"--", alpha=.6)
    ax[0].set_xlabel(r"$\alpha$"); ax[0].set_ylabel("MAD / CRLB")
    ax[0].set_title("Single (solid) vs two-component LS (dashed)"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=6)
    for df, mk in zip(dfs, ["o","s","^"]):
        rr = [r for r in rows if r["df"] == df]
        ax[1].plot([r["alpha"] for r in rr], [r["single_hu"] for r in rr], mk+"-", label=f"single Hu df={df}N")
        ax[1].plot([r["alpha"] for r in rr], [r["two_hu"] for r in rr], mk+"--", alpha=.6)
    ax[1].set_xlabel(r"$\alpha$"); ax[1].set_ylabel("MAD / CRLB")
    ax[1].set_title("Single (solid) vs two-component Huber (dashed)"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "multicomponent_alpha.png"), dpi=170)
    print("\nwrote results/multicomponent_alpha.json and .png")

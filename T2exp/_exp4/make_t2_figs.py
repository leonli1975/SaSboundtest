# -*- coding: utf-8 -*-
import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); DIR=os.path.dirname(HERE)
FIG=os.path.join(DIR,"..","paper","figures"); os.makedirs(FIG,exist_ok=True)

# ---- Fig 9: alpha
def fig_alpha():
    rows=list(csv.DictReader(open(os.path.join(DIR,"_verify","step2_alpha.csv"),encoding="utf-8-sig")))
    fig,ax=plt.subplots(1,2,figsize=(9.6,3.7))
    for ds,col,lab in [("MTDSP2025_HH_pure","C0","HH pure"),("MTDSP2025_VV_pure","C1","VV pure")]:
        rr=[r for r in rows if r["dataset"]==ds]
        rr.sort(key=lambda r: float(r["range_m"]))
        x=[float(r["range_m"])/1000 for r in rr]; y=[float(r["alpha_logecf"]) for r in rr]
        ax[0].plot(x,y,"o-",color=col,ms=3,lw=1,label=lab)
    ax[0].axhspan(1.0,1.9,color="0.85",alpha=.4,label="band 1.0-1.9")
    ax[0].axhline(1.0,color="k",ls=":",lw=1)
    ax[0].set_xlabel("range (km)"); ax[0].set_ylabel(r"$\alpha$")
    ax[0].set_title("MTDSP 2025 pure: alpha vs range")
    ax[0].set_ylim(0.9,2.05); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
    aq=json.load(open(os.path.join(DIR,"_verify","alpha_quick.json")))
    files=list(aq.keys()); y=[aq[f]["alpha"] for f in files]
    cols=["C0" if "HH" in f else "C1" for f in files]
    x=np.arange(len(files))
    ax[1].scatter(x,y,c=cols,s=40,zorder=3)
    for i,(xi,yi) in enumerate(zip(x,y)):
        ax[1].errorbar(xi,yi,yerr=[[yi-aq[files[i]]["alpha_p10"]],[aq[files[i]]["alpha_p90"]-yi]],
                       fmt="none",ecolor=cols[i],alpha=.5,capsize=2)
    ax[1].axhspan(1.0,1.9,color="0.85",alpha=.4)
    ax[1].axhline(1.0,color="k",ls=":",lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels([f.replace("MTDSP","").replace(".mat","") for f in files],rotation=60,fontsize=6)
    ax[1].set_ylabel(r"file-level $\alpha$")
    ax[1].set_ylim(0.9,2.05); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig9_t2_alpha.pdf")); plt.close(fig)
    print("fig9 done")

# ---- Fig 10: estimator on real clutter
def fig_est():
    d=json.load(open(os.path.join(HERE,"exp3_rerun.json")))
    fig,ax=plt.subplots(1,2,figsize=(9.6,3.7))
    for pol,col,mk in [("HH","C0","o"),("VV","C1","s")]:
        for est,ls in [("LS","-"),("Huber","--")]:
            rr=[r for r in d if r["pol"]==pol and r["est"]==est]; rr.sort(key=lambda r:r["snr_db"])
            x=[r["snr_db"] for r in rr]
            ax[0].errorbar(x,[r["mad"] for r in rr],
                           yerr=[[r["mad"]-r["mad_lo"] for r in rr],[r["mad_hi"]-r["mad"] for r in rr]],
                           fmt=mk+ls,color=col,ms=4,capsize=2,label=f"{pol} {est}")
            ax[1].semilogy(x,[max(r["rmse"],1e-3) for r in rr],mk+ls,color=col,ms=4,label=f"{pol} {est}")
    ax[0].axhline(1.0,color="k",ls=":",lw=1,label="white CRLB")
    ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("MAD / white CRLB")
    ax[0].set_title("Typical error (MAD)"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=6,ncol=2)
    ax[1].axhline(1.0,color="k",ls=":",lw=1)
    ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("RMSE / white CRLB (log)")
    ax[1].set_title("MSE (outlier-dominated)"); ax[1].grid(alpha=.3,which="both")
    ax[1].annotate("HH +2 dB: 2.9% outliers",xy=(2.0,101),xytext=(2.5,300),fontsize=7,
                   arrowprops=dict(arrowstyle="->",lw=.8))
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig10_t2_estimator.pdf")); plt.close(fig)
    print("fig10 done")

if __name__=="__main__":
    fig_alpha(); fig_est()

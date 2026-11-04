# -*- coding: utf-8 -*-
"""Add the real GELCT to the MTDSP T2 comparison (same x0 + target as exp3_rerun)."""
import os, sys, json
import numpy as np
import scipy.io as sio

HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.append(HERE)
sys.path.append(os.path.join(HERE,"..","..","gelct_lfm","src"))
import exp3_rerun_v2 as E
from gelct_core import estimate_k_full_window, estimate_initial_frequency

N=E.N; n=np.arange(N); PRF=E.PRF; K_TRUE=E.K_TRUE
RES=os.path.join(HERE,"..","results"); os.makedirs(RES,exist_ok=True)

def run(fn, snr=2.0, cells=8, blocks=13, seed=12345):
    rng=np.random.default_rng(seed)
    d=sio.loadmat(os.path.join(E.DIR,fn),squeeze_me=True,struct_as_record=False,
                  variable_names=["pc","clutter_cells","meta"])
    pc=d["pc"]; cc=np.atleast_1d(d["clutter_cells"]).astype(int)-1
    cc=cc[(cc>=0)&(cc<pc.shape[1])]
    idx=cc[np.linspace(0,cc.size-1,min(cells,cc.size)).astype(int)]
    rows=[]
    for cell in idx:
        v=pc[:,cell].astype(np.complex128); e=np.abs(v); alive=e>0.05*np.median(e)
        starts=[s for s in range(0,v.size-N,N//2) if alive[s:s+N].all()][:blocks]
        for st in starts:
            x0=v[st:st+N]; sig=E.sigma_med(x0)
            A=np.sqrt(10**(snr/10)*sig**2)
            f0=rng.uniform(0.10,0.30)*PRF; k=K_TRUE*(1+rng.uniform(-0.05,0.05))
            s=A*E.chirp(n,f0,k,rng.uniform(0,2*np.pi)); x=x0+s
            k0=estimate_k_full_window(x,PRF)
            fg,_=estimate_initial_frequency(x,k0,PRF)
            fl,kl=E.ls_estimate(x,n)
            fh,kh=E.huber_estimate(x,n,(fl,kl))
            crlf,_=E.crlb(10**(snr/10),n)
            rows.append(dict(file=fn,snr=snr,df_g=float(fg-f0),df_ls=float(fl-f0),
                             df_hu=float(fh-f0),crlf=crlf))
    return rows

if __name__=="__main__":
    out=[]; summary={}
    for fn in E.FILES:
        rows=run(fn,snr=2.0)
        out+=rows
        g=np.array([r["df_g"] for r in rows]); l=np.array([r["df_ls"] for r in rows]); h=np.array([r["df_hu"] for r in rows])
        cr=np.array([r["crlf"] for r in rows])
        row=dict(file=fn,n=len(rows),
                 g=dict(mad=float(np.median(np.abs(g))/np.median(cr)),
                        rmse=float(np.sqrt(np.mean(g**2))),
                        out=float(np.mean(np.abs(g)/cr>10)*100)),
                 ls=dict(mad=float(np.median(np.abs(l))/np.median(cr)),
                         rmse=float(np.sqrt(np.mean(l**2))),
                         out=float(np.mean(np.abs(l)/cr>10)*100)),
                 hu=dict(mad=float(np.median(np.abs(h))/np.median(cr)),
                         rmse=float(np.sqrt(np.mean(h**2))),
                         out=float(np.mean(np.abs(h)/cr>10)*100)))
        summary[fn]=row
        print(f"{fn:26s} n={row['n']:3d} GELCT MAD/CRLB={row['g']['mad']:.2f} RMSE={row['g']['rmse']:.1f} out={row['g']['out']:.1f}% | LS {row['ls']['mad']:.2f}/{row['ls']['rmse']:.1f}/{row['ls']['out']:.1f}% | Huber {row['hu']['mad']:.2f}/{row['hu']['rmse']:.1f}/{row['hu']['out']:.1f}%",flush=True)
    json.dump(summary,open(os.path.join(RES,"exp6_gelct_real.json"),"w"),indent=1)
    print("wrote results/exp6_gelct_real.json")

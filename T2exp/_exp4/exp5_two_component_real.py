# -*- coding: utf-8 -*-
"""P0-1: two-component fit on the real MTDSP outlier rows (oracle-init two-component)."""
import os, sys, json
import numpy as np
import scipy.io as sio

HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(os.path.join(HERE,"..","..","sim"))
import exp3_rerun_v2 as E
from multicomponent_efficiency import ls_two, ls_two_obj

RES=os.path.join(HERE,"..","results"); os.makedirs(RES,exist_ok=True)
N=E.N; n=np.arange(N); PRF=E.PRF

def norm(f0,k): return f0/PRF, k/(PRF*PRF)
def phys(q): return q[0]*PRF, q[1]*PRF*PRF

def load_pc(fn):
    d=sio.loadmat(os.path.join(E.DIR,fn),squeeze_me=True,struct_as_record=False,
                  variable_names=["pc","clutter_cells"])
    return d["pc"]

def two_fit(x,f_s,k_s,f0,k0):
    a=norm(f_s,k_s); b=norm(f0,k0)
    fits=[]
    for init in [(a[0],a[1],b[0],b[1]),(b[0],b[1],a[0],a[1])]:
        ft=ls_two(x,N,*init)
        r=ls_two_obj(ft,x,N); fits.append((r,ft))
    r,ft=min(fits,key=lambda z:z[0])
    cand=[(ft[0],ft[1]),(ft[2],ft[3])]
    d=[(c[0]-b[0])**2+(c[1]-b[1])**2 for c in cand]
    i=int(np.argmin(d)); other=1-i
    return phys(cand[i]), phys(cand[other]), r

if __name__=="__main__":
    rows=json.load(open(os.path.join(HERE,"exp3_rerun_v2_rows.json")))
    ls_rows=[r for r in rows if r["est"]=="LS"]
    out=[r for r in ls_rows if abs(r["df"])>5.0]
    ctl=[r for r in ls_rows if abs(r["df"])<0.5][:15]
    cache={}; res=[]
    for tag,sel in [("outlier",out),("control",ctl)]:
        for r in sel:
            fn=r["file"]
            if fn not in cache: cache[fn]=load_pc(fn)
            pc=cache[fn]
            cell=int(r["cell"])-1; st=int(r["block"])
            x0=pc[st:st+N,cell].astype(np.complex128)
            sig=E.sigma_med(x0); A=np.sqrt(10**(r["snr_db"]/10)*sig**2)
            f0=float(r["f0_true"]); k0=float(r["k_true"])
            s=A*E.chirp(n,f0,k0,0.3); x=x0+s
            f_s,k_s=E.ls_estimate(x,n)
            (f_t,k_t),(f_c,k_c),_=two_fit(x,f_s,k_s,f0,k0)
            crlf,_=E.crlb(10**(r["snr_db"]/10),n)
            res.append(dict(tag=tag,file=fn,cell=int(r["cell"]),block=st,snr_db=r["snr_db"],
                            f0_true=f0,k_true=k0,
                            df_single=float(f_s-f0),df_two=float(f_t-f0),
                            df_clutter=float(f_c-f0),crlf=crlf,
                            r_single=abs(f_s-f0)/crlf,r_two=abs(f_t-f0)/crlf))
            print(f"{tag:7s} {fn:24s} c{int(r['cell']):2d} b{st:5d} snr{r['snr_db']:+4.0f} "
                  f"|df| single={abs(f_s-f0):8.1f} Hz  two={abs(f_t-f0):8.2f} Hz  CRLB={crlf:.2f} Hz",flush=True)
    json.dump(res,open(os.path.join(RES,"exp5_two_component_real.json"),"w"),indent=1)
    for tag in ("outlier","control"):
        g=[r for r in res if r["tag"]==tag]
        if g:
            rs=np.median([r["r_single"] for r in g]); rt=np.median([r["r_two"] for r in g])
            print(f"\n{tag}: n={len(g)}  median |df|/CRLB  single={rs:.1f}  two={rt:.1f}",flush=True)
    print("wrote results/exp5_two_component_real.json")

# -*- coding: utf-8 -*-
"""100-trial LS/Huber re-run on real clutter (no SαS-MLE, no mtdsp dependency)."""
import os, json, time
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize

DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRF=2000.0; N=256; NF=2048
K_SWEEP=80.0; T_CPI=N/PRF; K_TRUE=K_SWEEP/T_CPI

def chirp(n,f0,k,phi,fs=PRF):
    t=n/fs; return np.exp(1j*(2*np.pi*f0*t+np.pi*k*t*t+phi))
def to_phys(q): return q[0]*PRF, q[1]*PRF/T_CPI
def to_norm(f0,k): return [f0/PRF, k*T_CPI/PRF]
def prof_ls(x,f0,k,n):
    s=chirp(n,f0,k,0.0); c=np.vdot(s,x); return abs(c)**2, c/max(n.size,1)
def ls_estimate(x,n,k0=K_TRUE,refine=True):
    ks=k0+np.linspace(-(4*K_TRUE+400),4*K_TRUE+400,25); best=(-1.0,0,0); w=np.hanning(n.size)
    for k in ks:
        y=x*np.exp(-1j*np.pi*k*(n/PRF)**2)*w
        F=np.abs(np.fft.fft(y,NF))**2; i=int(F.argmax())
        f0=(i if i<NF//2 else i-NF)*PRF/NF
        if F[i]>best[0]: best=(F[i],f0,k)
    _,f0,k=best
    if not refine: return f0,k
    obj=lambda q:-prof_ls(x,*to_phys(q),n)[0]
    r=minimize(obj,to_norm(f0,k),method="Nelder-Mead",options=dict(xatol=1e-7,fatol=1e-3,maxiter=600))
    return to_phys(r.x)
def huber_estimate(x,n,init,c=1.345,iters=10):
    s0=chirp(n,*to_phys(to_norm(*init)),0.0); a0=np.mean(s0.conj()*x)
    sc=1.4826*np.median(np.abs(x-a0*s0))+1e-12
    def loss(q):
        p=to_phys(q); s=chirp(n,p[0],p[1],0.0); w=np.ones(n.size)
        for _ in range(iters):
            a=np.sum(w*x*s.conj())/max(np.sum(w),1e-12); u=np.abs(x-a*s)/sc
            w=np.where(u<=c,1.0,c/np.maximum(u,1e-12))
        a=np.sum(w*x*s.conj())/max(np.sum(w),1e-12); u=np.abs(x-a*s)/sc
        return float(np.sum(np.where(u<=c,0.5*u**2,c*(u-0.5*c))))
    r=minimize(loss,to_norm(*init),method="Nelder-Mead",options=dict(xatol=1e-7,fatol=1e-6,maxiter=400))
    return to_phys(r.x)
def crlb(snr,n,fs=PRF):
    t=n/fs
    J=2*snr*np.array([[(2*np.pi)**2*np.sum(t**2),(2*np.pi)*np.pi*np.sum(t**3),2*np.pi*np.sum(t)],
                      [(2*np.pi)*np.pi*np.sum(t**3),np.pi**2*np.sum(t**4),np.pi*np.sum(t**2)],
                      [2*np.pi*np.sum(t),np.pi*np.sum(t**2),n.size]])
    C=np.linalg.inv(J)[:2,:2]
    return float(np.sqrt(abs(C[0,0]))), float(np.sqrt(abs(C[1,1])))
def sigma_med(v): return float(np.median(np.abs(v))/1.1774+1e-12)

def run(fn, snrs=(2.0,6.0,10.0), cells=8, blocks=13, rng=None):
    d=sio.loadmat(os.path.join(DIR,fn),squeeze_me=True,struct_as_record=False,
                  variable_names=["pc","clutter_cells","meta"])
    pc=d["pc"]; m=d["meta"]; pol=str(getattr(m,"polarization",""))
    cc=np.atleast_1d(d["clutter_cells"]).astype(int)-1
    cc=cc[(cc>=0)&(cc<pc.shape[1])]
    idx=cc[np.linspace(0,cc.size-1,min(cells,cc.size)).astype(int)]
    n=np.arange(N); rows=[]
    for cell in idx:
        v=pc[:,cell].astype(np.complex128); e=np.abs(v); alive=e>0.05*np.median(e)
        starts=[s for s in range(0,v.size-N,N//2) if alive[s:s+N].all()][:blocks]
        for st in starts:
            x0=v[st:st+N]; sig=sigma_med(x0)
            for snr in snrs:
                A=np.sqrt(10**(snr/10)*sig**2)
                f0=rng.uniform(0.10,0.30)*PRF; k=K_TRUE*(1+rng.uniform(-0.05,0.05))
                s=A*chirp(n,f0,k,rng.uniform(0,2*np.pi)); x=x0+s
                e_ls=ls_estimate(x,n); e_hu=huber_estimate(x,n,e_ls)
                for tag,(fe,ke) in (("LS",e_ls),("Huber",e_hu)):
                    rows.append(dict(file=fn,pol=pol,cell=int(cell)+1,block=int(st),snr_db=snr,
                                     est=tag,df=float(fe-f0),dk=float(ke-k)))
    return rows

def summ(rows,path):
    out=[]
    for (fn,est,snr),g in __import__("itertools").groupby(sorted(rows,key=lambda r:(r["file"],r["est"],r["snr_db"])),key=lambda r:(r["file"],r["est"],r["snr_db"])):
        g=list(g); df=np.array([r["df"] for r in g]); cr,_=crlb(10**(snr/10),np.arange(N))
        r=df/cr; outfrac=float(np.mean(np.abs(r)>10))
        rng=np.random.default_rng(0); boot=[]
        for _ in range(2000):
            b=rng.choice(r,r.size,replace=True); boot.append(np.median(np.abs(b)))
        lo,hi=np.percentile(boot,[2.5,97.5])
        out.append(dict(file=fn,pol=g[0]["pol"],est=est,snr_db=snr,n=int(df.size),
                        crlb_f=cr, mad=float(np.median(np.abs(r))),mad_lo=float(lo),mad_hi=float(hi),
                        rmse=float(np.sqrt(np.mean(r**2))),outlier_frac=outfrac,
                        trim5=float(np.mean(np.sort(r**2)[:max(1,int(0.95*r.size))]))))
        print(f"{fn:26s} {est:5s} {snr:+4.0f}dB n={df.size:3d} MAD={out[-1]['mad']:.3f} "
              f"[{lo:.3f},{hi:.3f}] RMSE={out[-1]['rmse']:.2f} out={outfrac:.1%} trim5={out[-1]['trim5']:.3f}")
    json.dump(out,open(path,"w"),indent=1)

if __name__=="__main__":
    rng=np.random.default_rng(7); allrows=[]
    for fn in ["MTDSP2022_HH_stare.mat","MTDSP2022_VV_stare.mat"]:
        t=time.time(); allrows+=run(fn,rng=rng); print(f"{fn} {time.time()-t:.1f}s")
    _here=os.path.dirname(os.path.abspath(__file__))
    json.dump(allrows,open(os.path.join(_here,"exp3_rerun_rows.json"),"w"))
    summ(allrows,os.path.join(_here,"exp3_rerun.json"))

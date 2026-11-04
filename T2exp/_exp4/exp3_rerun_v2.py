# -*- coding: utf-8 -*-
"""R2/R3: outlier mechanism (multi-start + oracle init) and colored-CRLB factors."""
import os, json, time
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
from scipy.stats import beta

DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRF=2000.0; N=256; NF=2048; K_SWEEP=80.0; T_CPI=N/PRF; K_TRUE=K_SWEEP/T_CPI
FILES=["MTDSP2022_HH_stare.mat","MTDSP2022_VV_stare.mat",
       "MTDSP2025_HH_pure.mat","MTDSP2025_VV_pure.mat"]
DIAG_FILES=["MTDSP2022_HH_stare.mat","MTDSP2025_HH_pure.mat"]
SNRS=(2.0,6.0,10.0)

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
    return ls_refine(x,n,f0,k)
def ls_refine(x,n,f0,k):
    obj=lambda q:-prof_ls(x,*to_phys(q),n)[0]
    r=minimize(obj,to_norm(f0,k),method="Nelder-Mead",
               options=dict(xatol=1e-7,fatol=1e-3,maxiter=600))
    return to_phys(r.x)
def ls_multi(x,n):
    best=(-1.0,0,0)
    for kk in (K_TRUE*0.95,K_TRUE,K_TRUE*1.05):
        f0,k=ls_estimate(x,n,k0=kk,refine=False)
        v=prof_ls(x,f0,k,n)[0]
        if v>best[0]: best=(v,f0,k)
    return ls_refine(x,n,best[1],best[2])
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
    r=minimize(loss,to_norm(*init),method="Nelder-Mead",
               options=dict(xatol=1e-7,fatol=1e-6,maxiter=400))
    return to_phys(r.x)
def crlb(snr,n,fs=PRF):
    t=n/fs
    J=2*snr*np.array([[(2*np.pi)**2*np.sum(t**2),(2*np.pi)*np.pi*np.sum(t**3),2*np.pi*np.sum(t)],
                      [(2*np.pi)*np.pi*np.sum(t**3),np.pi**2*np.sum(t**4),np.pi*np.sum(t**2)],
                      [2*np.pi*np.sum(t),np.pi*np.sum(t**2),n.size]])
    C=np.linalg.inv(J)[:2,:2]; return float(np.sqrt(abs(C[0,0]))),float(np.sqrt(abs(C[1,1])))
def sigma_med(v): return float(np.median(np.abs(v))/1.1774+1e-12)

def psd_factor(v,f0,fs=PRF,nfft=256):
    nb=v.size//nfft
    if nb<2: return 1.0
    P=np.zeros(nfft); m=min(nb,64)
    for b in range(m):
        seg=v[b*nfft:(b+1)*nfft]*np.hanning(nfft)
        P+=np.abs(np.fft.fftshift(np.fft.fft(seg)))**2
    P/=m; f=np.fft.fftshift(np.fft.fftfreq(nfft,1/fs))
    return float(np.interp(f0,f,P)/(P.mean()+1e-30))

def run(fn,snrs,cells=8,blocks=13,rng=None,diag=False):
    d=sio.loadmat(os.path.join(DIR,fn),squeeze_me=True,struct_as_record=False,
                  variable_names=["pc","clutter_cells","meta"])
    pc=d["pc"]; m=d["meta"]; pol=str(getattr(m,"polarization",""))
    cc=np.atleast_1d(d["clutter_cells"]).astype(int)-1; cc=cc[(cc>=0)&(cc<pc.shape[1])]
    idx=cc[np.linspace(0,cc.size-1,min(cells,cc.size)).astype(int)]
    n=np.arange(N); rows=[]
    for cell in idx:
        v=pc[:,cell].astype(np.complex128); e=np.abs(v); alive=e>0.05*np.median(e)
        starts=[s for s in range(0,v.size-N,N//2) if alive[s:s+N].all()][:blocks]
        for st in starts:
            x0=v[st:st+N]; sig=sigma_med(x0); crms=float(np.sqrt(np.mean(np.abs(x0)**2)))
            for snr in snrs:
                A=np.sqrt(10**(snr/10)*sig**2)
                f0=rng.uniform(0.10,0.30)*PRF; k=K_TRUE*(1+rng.uniform(-0.05,0.05))
                s=A*chirp(n,f0,k,rng.uniform(0,2*np.pi)); x=x0+s
                cf=psd_factor(x0,f0)
                e_ls=ls_estimate(x,n); e_hu=huber_estimate(x,n,e_ls)
                ests=[("LS",e_ls),("Huber",e_hu)]
                if diag and abs(snr-2.0)<1e-9:
                    ests.append(("Oracle",ls_refine(x,n,f0,k)))
                    ests.append(("Multi",ls_multi(x,n)))
                for tag,(fe,ke) in ests:
                    rows.append(dict(file=fn,pol=pol,cell=int(cell)+1,block=int(st),
                                     snr_db=float(snr),est=tag,f0_true=float(f0),k_true=float(k),
                                     df=float(fe-f0),dk=float(ke-k),
                                     clutter_rms=crms,colored_factor=float(cf)))
    return rows

def cp_ci(k,n,alpha=0.05):
    lo=0.0 if k==0 else beta.ppf(alpha/2,k,n-k+1)
    hi=1.0 if k==n else beta.ppf(1-alpha/2,k+1,n-k)
    return float(lo),float(hi)

def summ(rows,path):
    out=[]
    keys=sorted(set((r["file"],r["est"],r["snr_db"]) for r in rows))
    for fn,est,snr in keys:
        g=[r for r in rows if r["file"]==fn and r["est"]==est and abs(r["snr_db"]-snr)<1e-9]
        df=np.array([r["df"] for r in g]); cf=np.array([r["colored_factor"] for r in g])
        cr,_=crlb(10**(snr/10),np.arange(N)); rw=df/cr; rc=df/(cr*cf)
        outl=np.abs(rw)>10; kout=int(outl.sum()); n=len(g)
        lo,hi=cp_ci(kout,n)
        rng=np.random.default_rng(0); boot=[]
        for _ in range(2000):
            b=rng.choice(rw,rw.size,replace=True); boot.append(np.median(np.abs(b)))
        mlo,mhi=np.percentile(boot,[2.5,97.5])
        out.append(dict(file=fn,pol=g[0]["pol"],est=est,snr_db=snr,n=n,crlb_f=cr,
                        mad=float(np.median(np.abs(rw))),mad_lo=float(mlo),mad_hi=float(mhi),
                        mad_colored=float(np.median(np.abs(rc))),
                        rmse=float(np.sqrt(np.mean(rw**2))),
                        rmse_colored=float(np.sqrt(np.mean(rc**2))),
                        outlier_frac=kout/n,outlier_lo=lo,outlier_hi=hi,
                        trim5=float(np.mean(np.sort(rw**2)[:max(1,int(0.95*rw.size))])),
                        outlier_blocks=[(r["cell"],r["block"],round(r["clutter_rms"],2)) for r in g if abs(r["df"]/cr)>10]))
        print(f"{fn:24s} {est:6s} {snr:+4.0f} n={n:3d} MAD={out[-1]['mad']:.3f} "
              f"col={out[-1]['mad_colored']:.3f} RMSE={out[-1]['rmse']:.2f} "
              f"out={out[-1]['outlier_frac']:.1%} CI[{lo:.1%},{hi:.1%}]")
    json.dump(out,open(path,"w"),indent=1)

if __name__=="__main__":
    rng=np.random.default_rng(21); allrows=[]
    for fn in FILES:
        t=time.time(); allrows+=run(fn,SNRS,rng=rng,diag=(fn in DIAG_FILES))
        print(f"{fn} {time.time()-t:.1f}s")
    here=os.path.dirname(os.path.abspath(__file__))
    json.dump(allrows,open(os.path.join(here,"exp3_rerun_v2_rows.json"),"w"))
    summ(allrows,os.path.join(here,"exp3_rerun_v2.json"))

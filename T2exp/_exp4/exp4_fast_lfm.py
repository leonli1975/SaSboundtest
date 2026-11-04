# -*- coding: utf-8 -*-
"""R1-A': semi-synthetic fast-time LFM injection into 2025 _pure if_raw (60 MHz)."""
import os, json
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize

DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS=60e6; N=480; NF=2048
F0P=10e6; KP=2.5e12

def analytic(x):
    n=x.size; X=np.fft.fft(x); h=np.zeros(n); h[0]=1
    if n%2==0: h[n//2]=1; h[1:n//2]=2
    else: h[1:(n+1)//2]=2
    return np.fft.ifft(X*h)

def chirp(n,f0,k,phi): t=n/FS; return np.exp(1j*(2*np.pi*f0*t+np.pi*k*t*t+phi))
def prof(x,f0,k,n):
    s=chirp(n,f0,k,0.0); c=np.vdot(s,x); return abs(c)**2
def ls(x,n,k0=KP,kspan=0.8e12,refine=True):
    ks=k0+np.linspace(-kspan,kspan,25); best=(-1.0,0,0); w=np.hanning(n.size)
    for k in ks:
        y=x*np.exp(-1j*np.pi*k*(n/FS)**2)*w
        F=np.abs(np.fft.fft(y,NF))**2; i=int(F.argmax())
        f0=(i if i<NF//2 else i-NF)*FS/NF
        if F[i]>best[0]: best=(F[i],f0,k)
    _,f0,k=best
    if not refine: return f0,k
    return refine_at(x,n,f0,k)
def refine_at(x,n,f0,k):
    sc=[FS,1e12]
    q0=[f0/FS,k/1e12]
    obj=lambda q:-prof(x,q[0]*FS,q[1]*1e12,n)
    r=minimize(obj,q0,method="Nelder-Mead",options=dict(xatol=1e-7,fatol=1e-3,maxiter=600))
    return r.x[0]*FS, r.x[1]*1e12
def crlb(snr,n):
    t=n/FS
    J=2*snr*np.array([[(2*np.pi)**2*np.sum(t**2),(2*np.pi)*np.pi*np.sum(t**3),2*np.pi*np.sum(t)],
                      [(2*np.pi)*np.pi*np.sum(t**3),np.pi**2*np.sum(t**4),np.pi*np.sum(t**2)],
                      [2*np.pi*np.sum(t),np.pi*np.sum(t**2),n.size]])
    C=np.linalg.inv(J)[:2,:2]; return float(np.sqrt(abs(C[0,0]))),float(np.sqrt(abs(C[1,1])))
def sigma_med(v): return float(np.median(np.abs(v))/1.1774+1e-12)

def run(fn,snr=10.0,trials=100,win=0):
    d=sio.loadmat(os.path.join(DIR,fn),squeeze_me=True,struct_as_record=False,
                  variable_names=["if_raw","meta"])
    ifr=d["if_raw"]; print(fn,"if_raw",ifr.shape)
    rng=np.random.default_rng(5); n=np.arange(N); rows=[]
    for tr in range(trials):
        pulse=int(rng.integers(0,ifr.shape[0]))
        xa=analytic(ifr[pulse,:].astype(np.float64))
        x0=xa[win:win+N]; sig=sigma_med(x0)
        A=np.sqrt(10**(snr/10)*sig**2)
        f0=F0P*(1+rng.uniform(-0.01,0.01)); k=KP*(1+rng.uniform(-0.01,0.01))
        x=x0+A*chirp(n,f0,k,rng.uniform(0,2*np.pi))
        fe,ke=ls(x,n)
        fo,ko=refine_at(x,n,f0,k)
        crf,crk=crlb(10**(snr/10),n)
        rows.append(dict(pulse=pulse,snr_db=snr,f0_true=float(f0),k_true=float(k),
                         df=float(fe-f0),dk=float(ke-k),
                         df_oracle=float(fo-f0),dk_oracle=float(ko-k),
                         crlb_f=crf,crlb_k=crk))
    df=np.array([r["df"] for r in rows]); do=np.array([r["df_oracle"] for r in rows])
    crf=rows[0]["crlb_f"]
    out=np.abs(df/crf)>10
    print(f"  LS: MAD={np.median(np.abs(df/crf)):.3f} RMSE={np.sqrt(np.mean((df/crf)**2)):.2f} "
          f"out={out.mean()*100:.1f}%  | Oracle: MAD={np.median(np.abs(do/crf)):.3f} "
          f"RMSE={np.sqrt(np.mean((do/crf)**2)):.2f} out={(np.abs(do/crf)>10).mean()*100:.1f}%")
    json.dump(rows,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "exp4_fast_lfm_"+fn.replace('.mat','')+".json"),"w"),indent=1)

if __name__=="__main__":
    run("MTDSP2025_HH_pure.mat")
    run("MTDSP2025_VV_pure.mat")

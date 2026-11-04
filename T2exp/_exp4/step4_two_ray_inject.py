# -*- coding: utf-8 -*-
"""Synthetic two-ray injection: single-component LS on a two-component (direct+specular) chirp."""
import os, sys, json
import numpy as np
import scipy.io as sio
HERE=os.path.dirname(os.path.abspath(__file__)); DIR=os.path.dirname(HERE)
sys.path.insert(0,HERE)
from exp3_rerun import chirp, ls_estimate, crlb, sigma_med, K_TRUE, N, PRF

def main():
    d=sio.loadmat(os.path.join(DIR,"MTDSP2022_HH_stare.mat"),squeeze_me=True,
                  struct_as_record=False,variable_names=["pc","clutter_cells"])
    pc=d["pc"]; cc=np.atleast_1d(d["clutter_cells"]).astype(int)-1; cc=cc[(cc>=0)&(cc<pc.shape[1])]
    cell=cc[cc.size//2]; v=pc[:,cell].astype(np.complex128)
    alive=np.abs(v)>0.05*np.median(np.abs(v))
    st=next(s for s in range(0,v.size-N,N//2) if alive[s:s+N].all())
    x0=v[st:st+N]; sig=sigma_med(x0); n=np.arange(N)
    rng=np.random.default_rng(11)
    dfds=[0.005,0.01,0.02,0.05,0.1,0.5,1,2,5,7.8,10,20,50]
    rhos=[0.0,0.2,0.4,0.6]
    snr=10.0; A=np.sqrt(10**(snr/10)*sig**2); T=100
    rows=[]
    for rho in rhos:
        for dfd in dfds:
            dfs=[]
            for tr in range(T):
                f0=rng.uniform(0.10,0.30)*PRF; k=K_TRUE*(1+rng.uniform(-0.05,0.05))
                phi=rng.uniform(0,2*np.pi); phis=rng.uniform(0,2*np.pi)
                s=A*(chirp(n,f0,k,phi)+rho*chirp(n,f0+dfd,k,phi+phis))
                fe,ke=ls_estimate(x0+s,n)
                dfs.append(fe-f0)
            dfs=np.array(dfs); cr,_=crlb(10**(snr/10),n)
            rows.append(dict(rho=rho,dfd_hz=dfd,n=T,crlb_f=cr,
                             rms=float(np.sqrt(np.mean(dfs**2))),
                             mad=float(np.median(np.abs(dfs-np.median(dfs)))),
                             mean=float(np.mean(dfs))))
            print(f"rho={rho:.1f} dfD={dfd:6.3f} Hz  RMS={rows[-1]['rms']:7.3f} "
                  f"MAD={rows[-1]['mad']:7.3f} mean={rows[-1]['mean']:8.3f}  CRLB={cr:.3f}")
    json.dump(rows,open(os.path.join(HERE,"step4_two_ray_inject.json"),"w"),indent=1)
    print("resolution 1/T =",PRF/N,"Hz; measured dfD < 0.02 Hz")

if __name__=="__main__":
    main()

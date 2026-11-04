"""P0-2: P=3 / unequal amplitude / different chirp rate; block CRLB achievability."""
import json, os, sys, numpy as np
from scipy.optimize import minimize

HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE); sys.path.append(os.path.join(HERE,"..","..","paper1","code"))
from crlb_lfm import gram_single, gram_multi, crlb, Jmap
from m1_quant_corrected import csas

RES=os.path.join(HERE,"..","results"); os.makedirs(RES,exist_ok=True)
N=256; ALPHA=1.5; TRIALS=30; F1,K1=0.1,0.001; SCALE=0.5
DFS=[1.0,2.0,4.0]

def sig(N,f,k):
    n=np.arange(N); return np.exp(1j*(2*np.pi*f*n+np.pi*k*n*n))

def amps(e_list,x):
    m=len(e_list)
    A=np.array([[np.vdot(e_list[i],e_list[j]) for j in range(m)] for i in range(m)],dtype=complex)
    b=np.array([np.vdot(e_list[i],x) for i in range(m)],dtype=complex)
    return np.linalg.solve(A,b)

def ls_p_obj(p,x,N):
    m=len(p)//2
    e_list=[sig(N,p[2*i],p[2*i+1]) for i in range(m)]
    c=amps(e_list,x)
    r=x-sum(c[i]*e_list[i] for i in range(m))
    return float(np.real(np.vdot(r,r)))

def ls_p(x,N,comps):
    p0=np.array([v for c in comps for v in c],float)
    res=minimize(ls_p_obj,p0,args=(x,N),method="Nelder-Mead",
                 options={"xatol":1e-10,"fatol":1e-13,"maxiter":300})
    p=res.x
    return [(p[2*i],p[2*i+1]) for i in range(len(comps))], res.fun

def ls_single(x,N,f0,k0):
    res=minimize(lambda p: ls_p_obj(p,x,N),[f0,k0],method="Nelder-Mead",
                 options={"xatol":1e-11,"fatol":1e-14,"maxiter":400})
    return res.x

def mad_ci(v,B=1000,seed=0):
    v=np.asarray(v,float); rng=np.random.default_rng(seed)
    med=float(np.median(v))
    boots=np.array([np.median(rng.choice(v,size=v.size,replace=True)) for _ in range(B)])
    return med,float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))

CASES=[
    ("P3 equal, same k", [1.0,1.0,1.0], [K1,K1,K1]),
    ("P3 unequal amp",   [1.0,0.7,0.4], [K1,K1,K1]),
    ("P3 different k",   [1.0,1.0,1.0], [K1,0.8*K1,1.2*K1]),
]

if __name__=="__main__":
    G1=gram_single(N,F1,K1); C1,_=crlb(G1,Jmap[ALPHA],[0,1])
    cf1=float(np.sqrt(C1[0,0]))
    out=[]
    for name,amps3,ks3 in CASES:
        rows=[]
        for df in DFS:
            fs=[F1,F1+df/N,F1+2*df/N]
            comps=[(fs[i],ks3[i],amps3[i],0.0) for i in range(3)]
            Gm=gram_multi(N,comps); C3,_=crlb(Gm,Jmap[ALPHA],[0,1])
            cf3=float(np.sqrt(C3[0,0]))
            es,ep=[],[]
            for t in range(TRIALS):
                rng=np.random.default_rng(900000+int(100*df)+t+int(1000*sum(amps3)))
                x=sum(amps3[i]*sig(N,fs[i],ks3[i]) for i in range(3))+SCALE*csas(N,ALPHA,rng)
                fs_hat=ls_single(x,N,F1,K1)
                p3,_=ls_p(x,N,[(fs[i],ks3[i]) for i in range(3)])
                es.append(abs(fs_hat[0]-F1)/cf1)
                ep.append(abs(p3[0][0]-F1)/cf3)
            ms=mad_ci(es,seed=11+int(10*df)); mp=mad_ci(ep,seed=12+int(10*df))
            rows.append(dict(df=df,cf1=cf1,cf3=cf3,ratio=cf3/cf1,
                             single=dict(mad=ms[0],lo=ms[1],hi=ms[2]),
                             p3=dict(mad=mp[0],lo=mp[1],hi=mp[2])))
            print(f"{name:22s} df={df:4.1f}N  single={ms[0]:7.2f} [{ms[1]:.2f},{ms[2]:.2f}]  "
                  f"P3={mp[0]:5.2f} [{mp[1]:.2f},{mp[2]:.2f}]  std_ratio={cf3/cf1:.2f}", flush=True)
        out.append(dict(case=name,amps=amps3,ks=ks3,rows=rows))
    json.dump(out,open(os.path.join(RES,"unified_multi_p3.json"),"w"),indent=1)
    print("\nwrote results/unified_multi_p3.json")

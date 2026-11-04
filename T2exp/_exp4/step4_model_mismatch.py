# -*- coding: utf-8 -*-
"""Step 4: model mismatch on real sea clutter.
A) two-ray geometry (predicted fast-time / slow-time bias)
B) real clutter slow-time spectrum and coherent-block outliers
C) compare with the observed exp3 MAD/RMSE
"""
import os, json
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.dirname(HERE)
C = 299792458.0
FC = 9.4e9
LAM = C/FC
H = {"MTDSP2019_HH_scan":80.0, "MTDSP2020_HH_stare":25.0,
     "MTDSP2022_HH_stare":80.0, "MTDSP2025_HH_pure":80.0,
     "MTDSP2025_VV_pure":80.0}
SHIP_H = 10.0

def two_ray(h, R, ht, k, vr=5.0):
    dR = 2.0*h*ht/R
    dtau = dR/C
    fb = k*dtau
    dfs = (2.0/LAM)*(h*ht/R**2)*vr
    return dR, dtau, fb, dfs

def main():
    rows = []
    for fn in ["MTDSP2019_HH_scan.mat","MTDSP2020_HH_stare.mat",
               "MTDSP2022_HH_stare.mat","MTDSP2025_HH_pure.mat"]:
        d = sio.loadmat(os.path.join(DIR,fn), squeeze_me=True,
                        struct_as_record=False,
                        variable_names=["meta","range_m","grazing_deg"])
        m = d["meta"]; r = np.asarray(d["range_m"]); g = np.asarray(d["grazing_deg"])
        h = H[os.path.splitext(fn)[0]]
        k = float(getattr(m,"chirp_rate_hz_s",np.nan))
        wh = float(getattr(m,"wave_height_m",0.5))
        if wh != wh or wh <= 0: wh = 0.5
        R = float(np.median(r))
        ps = np.arcsin(min(1.0,h/R)) if not np.isfinite(g).any() else np.nanmedian(g)
        for ht, tag in [(SHIP_H,"ship"), (wh,"wave")]:
            dR,dtau,fb,dfs = two_ray(h,R,ht,k)
            rows.append(dict(file=fn, height_m=h, range_med_m=R, grazing_deg=float(ps),
                             target=tag, target_h_m=ht, dR_m=float(dR), dtau_ns=float(dtau*1e9),
                             fast_beat_hz=float(fb), doppler_sep_hz=float(dfs)))
            print(f"{fn:32s} {tag:5s} h={h:4.0f} R={R:6.0f} psi={ps:.3f} "
                  f"dR={dR:.3f}m dtau={dtau*1e9:.3f}ns f_b={fb:.3f}Hz df_D={dfs:.3f}Hz")
    json.dump(rows, open(os.path.join(HERE,"step4_two_ray.json"),"w"), indent=1)

    # B) real clutter slow-time spectrum + coherent-block outliers
    d = sio.loadmat(os.path.join(DIR,"MTDSP2022_HH_stare.mat"), squeeze_me=True,
                    struct_as_record=False,
                    variable_names=["pc","clutter_cells","meta"])
    pc = d["pc"]; cc = np.atleast_1d(d["clutter_cells"]).astype(int)-1
    cc = cc[(cc>=0)&(cc<pc.shape[1])]
    N = 256
    idx = cc[np.linspace(0,cc.size-1,min(16,cc.size)).astype(int)]
    spec = []
    for cell in idx:
        v = pc[:,cell].astype(np.complex128)
        v = v - v.mean()
        # Welch-like: average |FFT|^2 over 256-pulse blocks
        nb = v.size//N
        if nb < 2: continue
        P = np.zeros(N)
        for b in range(nb):
            seg = v[b*N:(b+1)*N]*np.hanning(N)
            P += np.abs(np.fft.fftshift(np.fft.fft(seg)))**2
        P /= nb
        f = np.fft.fftshift(np.fft.fftfreq(N, d=1/2000.0))
        spec.append((f,P))
    P = np.mean([s[1] for s in spec],axis=0); f=spec[0][0]
    Pn = P/P.mean()
    band = (np.abs(f)>=200)&(np.abs(f)<=600)
    print(f"\nclutter PSD at 200-600 Hz: {10*np.log10(Pn[band].mean()):.1f} dB (rel. mean)")
    print(f"clutter PSD at DC: {10*np.log10(Pn[np.abs(f)<10].mean()):.1f} dB")
    # coherent-block measure: block-to-block complex correlation of the clutter
    v = pc[:,idx[0]].astype(np.complex128); v=v-v.mean()
    nb=v.size//N; segs=[v[b*N:(b+1)*N] for b in range(nb)]
    cc_meas=[]
    for b in range(nb-1):
        a=segs[b]; c=segs[b+1]
        cc_meas.append(abs(np.vdot(a,c))/(np.linalg.norm(a)*np.linalg.norm(c)+1e-12))
    print(f"block-to-block coherence median={np.median(cc_meas):.3f}")
    json.dump(dict(psd_200_600_db=float(10*np.log10(Pn[band].mean())),
                   psd_dc_db=float(10*np.log10(Pn[np.abs(f)<10].mean())),
                   block_coherence_med=float(np.median(cc_meas)),
                   n_blocks=int(nb)),
              open(os.path.join(HERE,"step4_clutter.json"),"w"), indent=1)

    # observed exp3 numbers at +2 dB (from exp3_summary.csv)
    import csv
    obs=[]
    with open(os.path.join(DIR,"_exp3","exp3_summary.csv"),encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["src"].startswith("MTDSP2022") and abs(float(row["snr_db"])-2.0)<1e-6:
                obs.append(row)
    for r_ in obs:
        print(f"observed {r_['src']} {r_['est']}: MAD={float(r_['mad_f0']):.3f} Hz "
              f"RMSE={float(r_['rmse_f0']):.1f} Hz")
    json.dump(obs, open(os.path.join(HERE,"step4_observed.json"),"w"), indent=1)

if __name__=="__main__":
    main()

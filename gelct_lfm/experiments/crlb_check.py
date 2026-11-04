import os, sys, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE,"..","src")); sys.path.append(os.path.join(HERE,"..","..","sim"))
from crlb_lfm import gram_single, crlb, Jmap
from sas_noise import kappa_alpha, crlb_chirp_rate, crlb_initial_frequency
N, FS, TS = 1024, 10240.0, 0.1
FT, KT = 1000.0, 10000.0
FN, KN = FT/FS, KT/FS**2
print(f"N={N} FS={FS} FN={FN:.6f} KN={KN:.3e} 1/FS={1/FS:.3e}")
for alpha in [1.0,1.5,2.0]:
    G = gram_single(N, FN, KN)
    C,_ = crlb(G, Jmap[alpha] if alpha in Jmap else Jmap[1.10], [0,1])
    cf, ck = float(np.sqrt(C[0,0])), float(np.sqrt(C[1,1]))
    kappa = kappa_alpha(alpha); gsnr=1.0
    crf_s = np.sqrt(crlb_initial_frequency(N, 1/FS, kappa, gsnr))
    crk_s = np.sqrt(crlb_chirp_rate(N, 1/FS, kappa, gsnr))
    crf_c = np.sqrt(crlb_initial_frequency(N, TS, kappa, gsnr))
    crk_c = np.sqrt(crlb_chirp_rate(N, TS, kappa, gsnr))
    print(f"\nalpha={alpha:.1f} kappa={kappa:.4f}")
    print(f"  crlb_lfm  cf_norm={cf:.4e}  ck_norm={ck:.4e}")
    print(f"  crlb_lfm  cf_phys={cf*FS:.4e} Hz  ck_phys={ck*FS**2:.4e} Hz/s")
    print(f"  sas ts=1/FS  std_f={crf_s:.4e}  std_k={crk_s:.4e}")
    print(f"  sas ts=0.1   std_f={crf_c:.4e}  std_k={crk_c:.4e}")
    print(f"  ratios lfm/sas(1/FS): f={cf*FS/crf_s:.3e}  k={ck*FS**2/crk_s:.3e}")
    print(f"  ratios lfm/sas(0.1) : f={cf*FS/crf_c:.3e}  k={ck*FS**2/crk_c:.3e}")

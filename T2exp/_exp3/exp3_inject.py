# -*- coding: utf-8 -*-
r"""Step 3：向真实海杂波距离单元注入已知 LFM（慢时间 chirp），比较 LS / Huber / SαS-MLE。

模型：x[n] = clutter[n] + A·exp(j(2π f0 t_n + π k t_n² + φ))，t_n = n/PRF，n=0..N-1
待估：f0（多普勒）、k（多普勒变化率）；A、φ 为冗余参数（内部按 LS 消去）。
噪声：① 真实海杂波（逐单元、逐块）；② 同 α 的合成 SαS；③ 供对照的复高斯。
参考：高斯 CRLB（真实海杂波 α<2 时不存在有限方差，仅作对照）。

用法： python exp3_inject.py --quick      （小规模自检）
       python exp3_inject.py              （完整实验，约 10~20 分钟）
"""
import os, sys, json, csv, time
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize

ROOT = r"F:\海杂波数据"
DIR = os.path.join(ROOT, "T2exp")
OUTD = os.path.join(DIR, "_exp3")
os.makedirs(OUTD, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, r"MTDSP202501\mtdsp_tools"))
from mtdsp import clutter as C

PRF = 2000.0
N = 256                     # CPI 脉冲数
NF = 2048                   # 粗搜 FFT 长度（零填充）
K_SWEEP = 80.0              # 一个 CPI 内扫频宽度 (Hz)：k = K_SWEEP / T
T_CPI = N / PRF
K_TRUE = K_SWEEP / T_CPI    # Hz/s

# ---------------------------------------------------------------- 信号/工具


def chirp(n, f0, k, phi, fs=PRF):
    t = n / fs
    return np.exp(1j * (2 * np.pi * f0 * t + np.pi * k * t * t + phi))


def to_phys(q):
    """q=[f0/PRF, k*T/PRF] -> (f0 Hz, k Hz/s)"""
    return q[0] * PRF, q[1] * PRF / T_CPI


def to_norm(f0, k):
    return [f0 / PRF, k * T_CPI / PRF]


def prof_ls(x, f0, k, n):
    """给定 (f0,k) 时 A、φ 的 LS 解 -> 返回 |Σ x conj(s)|²（s 单位幅度）与 s。"""
    s = chirp(n, f0, k, 0.0)
    c = np.vdot(s, x)                       # Σ conj(s) x
    return abs(c) ** 2, c / max(n.size, 1)   # c/N = 复幅度 A e^{jφ}


def ls_estimate(x, n, k0=K_TRUE, kspan=None, refine=True):
    """LS（chirp 周期图）：k 粗扫 + FFT 找 f0 峰，再 Nelder-Mead 细化。"""
    if kspan is None:
        kspan = 4.0 * K_TRUE + 400.0
    ks = k0 + np.linspace(-kspan, kspan, 25)
    best = (-1.0, 0.0, 0.0)
    w = np.hanning(n.size)
    for k in ks:
        y = x * np.exp(-1j * np.pi * k * (n / PRF) ** 2) * w
        F = np.abs(np.fft.fft(y, NF)) ** 2
        i = int(F.argmax())
        f0 = (i if i < NF // 2 else i - NF) * PRF / NF
        if F[i] > best[0]:
            best = (F[i], f0, k)
    _, f0, k = best
    if not refine:
        return f0, k
    obj = lambda q: -prof_ls(x, *to_phys(q), n)[0]
    r = minimize(obj, to_norm(f0, k), method="Nelder-Mead",
                 options=dict(xatol=1e-7, fatol=1e-3, maxiter=600))
    return to_phys(r.x)


def huber_estimate(x, n, init, c=1.345, iters=10):
    """Huber-IRLS：尺度由初值残差定死后固定（不参与优化，否则会退化成放大尺度）。"""
    s0 = chirp(n, *to_phys(to_norm(*init)), 0.0)
    a0 = np.mean(s0.conj() * x)
    sc = 1.4826 * np.median(np.abs(x - a0 * s0)) + 1e-12

    def loss(q):
        p = to_phys(q)
        s = chirp(n, p[0], p[1], 0.0)
        w = np.ones(n.size)
        for _ in range(iters):
            a = np.sum(w * x * s.conj()) / max(np.sum(w), 1e-12)
            u = np.abs(x - a * s) / sc
            w = np.where(u <= c, 1.0, c / np.maximum(u, 1e-12))
        a = np.sum(w * x * s.conj()) / max(np.sum(w), 1e-12)
        u = np.abs(x - a * s) / sc
        return float(np.sum(np.where(u <= c, 0.5 * u ** 2, c * (u - 0.5 * c))))

    res = minimize(loss, to_norm(*init), method="Nelder-Mead",
                   options=dict(xatol=1e-7, fatol=1e-6, maxiter=400))
    return to_phys(res.x)


class SasLikelihood:
    """SαS 对数似然查表：log p_α(|r|)，γ=1，按 log|r| 线性插值。"""

    def __init__(self, alpha, xmin=1e-3, xmax=60.0, m=700):
        self.alpha = alpha
        self.x = np.logspace(np.log10(xmin), np.log10(xmax), m)
        p = C.sas_pdf(self.x, alpha, 1.0)
        p = np.maximum(p, 1e-300)
        self.logp = np.log(p)
        self.lx = np.log(self.x)
        # |X| 的中位数（γ=1），供 fit_gamma 用；只算一次
        # 对称 SαS：|X| 的中位数 = Q(0.75)（Q(0.5)=0，不能直接用）
        self.med1 = float(C.sas_quantile(np.array([0.75]), alpha)[0])

    def nll(self, r, gamma):
        u = np.abs(r) / gamma
        u = np.maximum(u, 1e-6)
        lu = np.log(u)
        lp = np.interp(lu, self.lx, self.logp, left=self.logp[0], right=np.nan)
        tail = lu > self.lx[-1]
        if np.any(tail):                      # 尾部按 |r|^{-(α+1)} 外推
            lp[tail] = self.logp[-1] - (self.alpha + 1.0) * (lu[tail] - self.lx[-1])
        return float(-np.sum(lp) + r.size * np.log(gamma))

    def fit_gamma(self, r):
        """用中位数定标（SαS 的 |X| 中位数 → γ）。"""
        med = np.median(np.abs(r)) + 1e-12
        return float((med / max(self.med1, 1e-12)) ** self.alpha)


def mle_estimate(x, n, lik, init):
    """SαS-MLE：γ 由初值残差定死后固定，(f0,k) 用 Nelder-Mead 最小化 -logL。"""
    s0 = chirp(n, *to_phys(to_norm(*init)), 0.0)
    a0 = np.mean(s0.conj() * x)
    g = lik.fit_gamma(x - a0 * s0)

    def nll(q):
        p = to_phys(q)
        s = chirp(n, p[0], p[1], 0.0)
        a = np.mean(s.conj() * x)
        return lik.nll(x - a * s, g)

    r = minimize(nll, to_norm(*init), method="Nelder-Mead",
                 options=dict(xatol=1e-7, fatol=1e-3, maxiter=400))
    return to_phys(r.x)


def crlb(snr, n, fs=PRF):
    """复高斯白噪声下 (f0,k) 的 CRLB：对 (f0,k,φ) 三参数 FIM 取 Schur 补（φ 为冗余相位）。"""
    t = n / fs
    J = 2 * snr * np.array([
        [(2*np.pi)**2 * np.sum(t**2), (2*np.pi)*np.pi * np.sum(t**3), 2*np.pi*np.sum(t)],
        [(2*np.pi)*np.pi * np.sum(t**3), np.pi**2 * np.sum(t**4),      np.pi*np.sum(t**2)],
        [2*np.pi*np.sum(t),              np.pi*np.sum(t**2),           n.size]])
    try:
        Cm = np.linalg.inv(J)[:2, :2]
        return float(np.sqrt(abs(Cm[0, 0]))), float(np.sqrt(abs(Cm[1, 1])))
    except np.linalg.LinAlgError:
        return np.nan, np.nan


def sigma_med(v):
    """复高斯等效标准差：median|v| = 1.1774σ（对重尾稳健）。"""
    return float(np.median(np.abs(v)) / 1.1774 + 1e-12)


# ---------------------------------------------------------------- 单个文件的实验
def run_file(fn, cells_n=8, blocks=8, snrs=(-10, -6, -2, 2, 6, 10), trials=2,
             alpha=None, rng=None, quick=False):
    d = sio.loadmat(fn, squeeze_me=True, struct_as_record=False,
                    variable_names=["pc", "clutter_cells", "meta", "range_m"])
    pc = d["pc"]
    cc = np.atleast_1d(d["clutter_cells"]).astype(int) - 1
    cc = cc[(cc >= 0) & (cc < pc.shape[1])]
    m = d["meta"]
    pol = str(getattr(m, "polarization", ""))
    if quick:
        cells_n, blocks, trials = 3, 3, 1
    idx = cc[np.linspace(0, cc.size - 1, min(cells_n, cc.size)).astype(int)]
    lik = SasLikelihood(alpha) if np.isfinite(alpha) else None
    n = np.arange(N)
    rows = []
    for ci, cell in enumerate(idx):
        v = pc[:, cell]
        e = np.abs(v)
        alive = e > 0.05 * np.median(e)
        starts = [s for s in range(0, v.size - N, N // 2) if alive[s:s + N].all()][:blocks]
        if not starts:
            continue
        for st in starts:
            x0 = v[st:st + N].astype(np.complex128)
            sig = sigma_med(x0)
            for snr in snrs:
                A = np.sqrt(10 ** (snr / 10) * sig ** 2)
                for tr in range(trials):
                    f0 = rng.uniform(0.10, 0.30) * PRF
                    k = K_TRUE * (1 + rng.uniform(-0.05, 0.05))
                    phi = rng.uniform(0, 2 * np.pi)
                    s = A * chirp(n, f0, k, phi)
                    x = x0 + s
                    t0 = time.time()
                    e_ls = ls_estimate(x, n) if not quick else ls_estimate(x, n, refine=False)
                    e_hu = huber_estimate(x, n, e_ls)
                    e_ml = mle_estimate(x, n, lik, e_ls) if lik is not None else (np.nan, np.nan)
                    for tag, (fe, ke) in (("LS", e_ls), ("Huber", e_hu), ("MLE", e_ml)):
                        rows.append(dict(file=os.path.basename(fn), pol=pol, alpha=alpha,
                                         cell=int(cell) + 1, block=int(st), snr_db=float(snr),
                                         trial=tr, est=tag,
                                         f0_true=float(f0), k_true=float(k),
                                         f0_hat=float(fe), k_hat=float(ke),
                                         df=float(fe - f0), dk=float(ke - k)))
            if quick and ci >= 1:
                break
    return rows


# ---------------------------------------------------------------- 合成数据
def run_synth(alpha, gamma, snrs, trials=6, rng=None, kind="sas"):
    n = np.arange(N)
    lik = SasLikelihood(alpha) if np.isfinite(alpha) else None
    rows = []
    for snr in snrs:
        for tr in range(trials):
            if kind == "sas":
                w = C.sas_quantile(rng.random(2 * N), alpha)
                w = (w[::2] + 1j * w[1::2]) / np.sqrt(2)
            else:
                w = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
            w = w * gamma
            sig = sigma_med(w)
            A = np.sqrt(10 ** (snr / 10) * sig ** 2)
            f0 = rng.uniform(0.10, 0.30) * PRF
            k = K_TRUE * (1 + rng.uniform(-0.05, 0.05))
            x = w + A * chirp(n, f0, k, rng.uniform(0, 2 * np.pi))
            e_ls = ls_estimate(x, n)
            e_hu = huber_estimate(x, n, e_ls)
            e_ml = mle_estimate(x, n, lik, e_ls) if lik is not None else (np.nan, np.nan)
            for tag, (fe, ke) in (("LS", e_ls), ("Huber", e_hu), ("MLE", e_ml)):
                rows.append(dict(file=f"synth_{kind}_a{alpha:.2f}", pol="-", alpha=alpha,
                                 cell=0, block=0, snr_db=float(snr), trial=tr, est=tag,
                                 f0_true=float(f0), k_true=float(k), f0_hat=float(fe),
                                 k_hat=float(ke), df=float(fe - f0), dk=float(ke - k)))
    return rows


def summarize(rows, path=None):
    out = []
    for (key, est, snr), g in _group(rows):
        df = np.array([r["df"] for r in g]); dk = np.array([r["dk"] for r in g])
        mad = float(np.median(np.abs(df - np.median(df))))
        mse = float(np.mean(np.abs(df) ** 2))
        k_ = max(1, int(round(0.05 * df.size)))
        trim = np.sort(np.abs(df) ** 2)[: df.size - k_]
        cr_f, cr_k = crlb(10 ** (snr / 10), np.arange(N))
        out.append(dict(src=key[0], pol=key[1], alpha=key[2], est=est, snr_db=snr,
                        crlb_f0=cr_f, crlb_k=cr_k, ratio_f0=float(np.sqrt(mse) / max(cr_f, 1e-12)),
                        n=len(g), bias_f0=float(np.mean(df)),
                        mad_f0=mad, rmse_f0=float(np.sqrt(mse)),
                        trim5_mse_f0=float(np.mean(trim)),
                        bias_k=float(np.mean(dk)), rmse_k=float(np.sqrt(np.mean(dk ** 2)))))
    if path:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader(); w.writerows(out)
    return out


def _group(rows):
    import collections
    d = collections.OrderedDict()
    for r in rows:
        k = ((r["file"], r["pol"], r["alpha"]), r["est"], r["snr_db"])
        d.setdefault(k, []).append(r)
    return d.items()


def main():
    quick = "--quick" in sys.argv
    rng = np.random.default_rng(20251)
    files = [("MTDSP2022_HH_stare.mat", None), ("MTDSP2022_VV_stare.mat", None)]
    aq = json.load(open(os.path.join(DIR, "_verify", "alpha_quick.json"), encoding="utf-8"))
    rows = []
    for fn, _ in files:
        a = aq.get(fn, {}).get("alpha", np.nan)
        print(f"== {fn}  α={a:.2f}", flush=True)
        r0 = run_file(os.path.join(DIR, fn), alpha=a, rng=rng, quick=quick)
        rows += r0
        print(f"   {len(r0)} 条估计", flush=True)
    if not quick:
        for a in (1.3, 1.6, 1.9):
            print(f"== 合成 SαS α={a}", flush=True)
            rows += run_synth(a, gamma=1.0, snrs=(-10, -6, -2, 2, 6, 10), trials=8, rng=rng)
    summ = summarize(rows, os.path.join(OUTD, "exp3_summary.csv"))
    json.dump(rows, open(os.path.join(OUTD, "exp3_rows.json"), "w", encoding="utf-8"),
              ensure_ascii=False, default=float)
    # 打印要点
    print("\n{:<28}{:<6}{:>5}{:>8}{:>12}{:>12}{:>12}".format(
        "数据", "估计", "SNR", "bias", "RMSE(f0)", "MAD", "trim5MSE"))
    for s in summ:
        if s["est"] == "LS" or abs(s["snr_db"] - 2) < 1e-6:
            print("{:<28}{:<6}{:>5.0f}{:>8.2f}{:>12.2f}{:>12.2f}{:>12.1f}".format(
                str(s["src"])[:27], s["est"], s["snr_db"], s["bias_f0"],
                s["rmse_f0"], s["mad_f0"], s["trim5_mse_f0"]))
    print("\n->", os.path.join(OUTD, "exp3_summary.csv"))


if __name__ == "__main__":
    main()

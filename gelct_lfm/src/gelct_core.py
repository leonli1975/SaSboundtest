# -*- coding: utf-8 -*-
"""
GELCT  --  Generalized Extended Linear Chirplet Transform
plus the LFM modulation-parameter estimator built on top of it.

Faithful implementation of Sections II-VI of

    Liu, M., Han, Y., Chen, Y., Song, H., Yang, Z., Gong, F. (2021).
    "Modulation Parameter Estimation of LFM Interference for Direct Sequence
     Spread Spectrum Communication System in Alpha-Stable Noise",
    IEEE Systems Journal, 15(1), 881-892.   DOI: 10.1109/JSYST.2020.2991078

    Eq. (1)   r(t) = x(t) + s(t) + e(t)
    Eq. (2)   s(t) = A exp(j 2 pi f0 t + j pi k t^2)
    Eq. (8)   f[r] = ( log_e(|r|^(1/e)) + 1 ) / |r| * r          (|r| != 0)
    Eq. (7)   G(t,w) = int f[r(tau)] h(tau-t) exp(-j w tau)
                       exp(-j 1/2 (tau-t)^2 (fs/(2 Ts)) tan(theta)) dtau
              theta_l = -pi/2 + l pi /(L+1),  l = 1..L
    Eq. (12)  R(u,v) = int int G(t,w) delta(t cos v + w sin v - u) dt dw
    Eq. (14)  k_hat  = -cot(v_hat) * df / dt
    Eq. (15)  r1(t)  = r(t) exp(-j pi k_hat t^2)
    Eq. (16)  GF(w)  = int f[r1(t)] exp(-j w t) dt
    Eq. (20)  f0_hat = argmax_f |GF| * df1/(2 pi)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sas_noise import E, complex_sas_noise, sas_rvs, gsnr_to_gamma


# ############################################################################
# 1.  Nonlinear transformation  (Eq. 8)
# ############################################################################
def nonlinear_transform(r: np.ndarray) -> np.ndarray:
    """Eq. (8):  f[r] = (log_e(|r|^(1/e)) + 1) / |r| * r ,   |r| != 0.

    log_e(|r|^(1/e)) = ln|r| / e, so the output modulus is |ln|r|/e + 1| and the
    phase of r is preserved exactly: f[r] = g(|r|) * r/|r| with a purely real
    gain g.  The enormous dynamic range of the impulsive S-alpha-S components is
    squeezed into a narrow logarithmic interval (Property 1, Appendix A), while
    the phase - and therefore the whole modulation information - is untouched.
    """
    r = np.asarray(r)
    a = np.abs(r)
    out = np.zeros_like(r, dtype=complex)
    nz = a > 0
    out[nz] = ((np.log(a[nz]) / E + 1.0) / a[nz]) * r[nz]
    return out


# ############################################################################
# 2.  Signal synthesis
# ############################################################################
@dataclass
class LFMParams:
    amplitude: float = 1.0
    f0: float = 1000.0            # initial frequency   [Hz]
    chirp_rate: float = 10000.0   # chirp rate k        [Hz/s]


def lfm_signal(n: int, fs: float, p: LFMParams) -> np.ndarray:
    """Sampled Eq. (2),  t = n/fs."""
    t = np.arange(n) / fs
    return p.amplitude * np.exp(1j * (2.0 * np.pi * p.f0 * t + np.pi * p.chirp_rate * t ** 2))


def dsss_signal(n: int, fs: float, chip_rate: float, power: float,
                rng: np.random.Generator) -> np.ndarray:
    """A simple BPSK DSSS useful signal x(t) (spreading + chip shaping)."""
    n_chips = max(1, int(round(n / fs * chip_rate)))
    chips = rng.choice([-1.0, 1.0], size=n_chips)
    idx = np.minimum((np.arange(n) / fs * chip_rate).astype(int), n_chips - 1)
    x = chips[idx].astype(complex)
    if n_chips > 1:
        m = max(3, int(fs / chip_rate)) // 2 * 2 + 1
        ker = np.hanning(m)
        x = np.convolve(x, ker / ker.sum(), mode="same")
    return x * np.sqrt(power / np.mean(np.abs(x) ** 2))


def received_signal(n: int, fs: float, p: LFMParams,
                    alpha: float = 1.5, gsnr_db: float = 0.0,
                    rng: np.random.Generator | None = None,
                    dsss: dict | None = None,
                    quadrature: str = "real",
                    amplitude: float | None = None):
    """Eq. (1) r(t) = x(t) + s(t) + e(t).

    GSNR convention of the paper, Eq. (6): the dispersion is fixed at gamma = 1
    and the generalised SNR is set through the signal power sigma_s^2 = A^2,
    i.e.  A = sqrt(10^(GSNR/10)).  Pass `amplitude` to fix A instead - then the
    dispersion is solved from Eq. (6).

    quadrature : 'real'    -> e(nTs) is a scalar S-alpha-S variable added to the
                              complex signal, exactly as written in Eq. (34);
                 'complex' -> independent S-alpha-S noise in both quadratures
                              (the model implied by Eq. (35)).
    """
    rng = rng or np.random.default_rng()
    if amplitude is None:
        amp = np.sqrt(10.0 ** (gsnr_db / 10.0))       # gamma stays 1
        gamma = 1.0
    else:
        amp = float(amplitude)
        gamma = gsnr_to_gamma(gsnr_db, amp ** 2)
    p = LFMParams(amp, p.f0, p.chirp_rate)

    s = lfm_signal(n, fs, p)
    if quadrature == "real":
        e = sas_rvs(alpha, gamma, n, rng).astype(complex)
    elif quadrature == "complex":
        e = complex_sas_noise(alpha, gamma, n, rng)
    else:
        raise ValueError(quadrature)

    if dsss is None:
        x = np.zeros(n, dtype=complex)
    else:
        x = dsss_signal(n, fs, dsss.get("chip_rate", 1024.0),
                        dsss.get("power", 0.1), rng)

    comp = {"x": x, "s": s, "e": e, "gamma": gamma, "alpha": alpha,
            "gsnr_db": gsnr_db, "amplitude": amp}
    return x + s + e, comp


# ############################################################################
# 3.  GELCT   (Eq. 7 / Eq. 21)
# ############################################################################
@dataclass
class GELCTConfig:
    win_len: int = 256          # analysis window                   [samples]
    hop: int = 8                # frame step                        [samples]
    nfft: int = 1024            # FFT length
    L: int = 32                 # number of chirp angles theta_l (Eq. 7)
    use_chirplet: bool = True   # False -> plain STFT of f[r] (baseline)
    one_sided: bool = True      # keep [0, fs/2)
    window: str = "hann"
    phase_2pi: bool = True      # read the Eq. (7) exponent as 2*pi*(cycles)


@dataclass
class GELCTResult:
    G: np.ndarray               # TF map (n_frames, n_freq)
    t_axis: np.ndarray          # frame-centre times          [s]
    f_axis: np.ndarray          # frequency-bin centres       [Hz]
    dt: float                   # hop / fs                    [s]
    df: float                   # fs / nfft                   [Hz]
    fs: float = 1.0             # sampling frequency          [Hz]
    n_samples: int = 0          # length of the analysed record
    cfg: GELCTConfig = field(repr=False, default=None)


def _window(kind: str, n: int) -> np.ndarray:
    if kind == "hann":
        return np.hanning(n)
    if kind == "hamming":
        return np.hamming(n)
    if kind == "gauss":
        return np.exp(-0.5 * ((np.arange(n) - (n - 1) / 2.0) / (n / 6.0)) ** 2)
    raise ValueError(kind)


def gelct(r: np.ndarray, fs: float, ts: float,
          cfg: GELCTConfig | None = None) -> GELCTResult:
    """Generalized Extended Linear Chirplet Transform, Eq. (7).

    Eq. (7) contains a chirp kernel exp(-j/2 (tau-t)^2 (fs/(2Ts)) tan(theta)).
    Substituting the kernel into the integral shows that the LFM ridge stays at
    w(t) = 2 pi f0 + 2 pi k t for *every* theta, while the residual quadratic
    phase inside the window, (2 pi k - c(theta)) (tau-t)^2/2, vanishes exactly
    when the kernel chirp matches the signal, c = 2 pi k.  The "extended" part
    of the transform therefore consists in adapting theta frame by frame: for
    each frame the theta_l that maximises the spectral concentration is kept,
    which turns the smeared ridge of a plain STFT into a sharp line.
    """
    r = np.asarray(r)
    cfg = cfg or GELCTConfig()
    n_total = r.size
    w = _window(cfg.window, cfg.win_len)
    rn = nonlinear_transform(r)

    starts = np.arange(0, max(1, n_total - cfg.win_len + 1), cfg.hop)
    if starts.size == 0:
        starts = np.array([0])
    n_frames = starts.size

    lag = (np.arange(cfg.win_len) - (cfg.win_len - 1) / 2.0) / fs   # [s]
    lag2 = lag ** 2

    l = np.arange(1, cfg.L + 1)
    theta = -np.pi / 2.0 + l * np.pi / (cfg.L + 1.0)                # Eq. (7)
    scale = 2.0 * np.pi if cfg.phase_2pi else 1.0
    c_grid = scale * (fs / (2.0 * ts)) * np.tan(theta)
    if not cfg.use_chirplet or cfg.L <= 1:
        c_grid = np.array([0.0])

    n_freq = cfg.nfft // 2 if cfg.one_sided else cfg.nfft
    G = np.zeros((n_frames, n_freq), dtype=complex)

    mods = [np.exp(-1j * 0.5 * c * lag2) if c != 0.0
            else np.ones(cfg.win_len, dtype=complex) for c in c_grid]

    for m, n0 in enumerate(starts):
        seg = rn[n0:n0 + cfg.win_len] * w
        if seg.size < cfg.win_len:
            seg = np.pad(seg, (0, cfg.win_len - seg.size))
        best, best_score = None, -np.inf
        for mod in mods:
            spec = np.fft.fft(seg * mod, cfg.nfft)[:n_freq]
            e = np.sum(np.abs(spec) ** 2)
            score = np.max(np.abs(spec) ** 2) / (e + 1e-300)
            if score > best_score:
                best_score, best = score, spec
        G[m, :] = best

    df = fs / cfg.nfft
    f_axis = np.arange(n_freq) * df
    t_axis = (starts + (cfg.win_len - 1) / 2.0) / fs
    return GELCTResult(G=G, t_axis=t_axis, f_axis=f_axis,
                       dt=cfg.hop / fs, df=df, fs=fs, n_samples=n_total, cfg=cfg)


# ############################################################################
# 4.  Radon transform and chirp-rate estimation   (Eq. 9-14)
# ############################################################################
def radon_sinogram(img: np.ndarray, n_angles: int = 361):
    """Classical discrete Radon transform, Eq. (12):

        R(u,v) = int int G(t,w) delta(t cos v + w sin v - u) dt dw

    img[i,j] : i = time bin, j = frequency bin.   ->  (R, v_axis, u_axis)
    """
    nt, nf = img.shape
    ii, jj = np.meshgrid(np.arange(nt), np.arange(nf), indexing="ij")
    v_axis = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    u_max = int(np.ceil(np.hypot(nt, nf))) + 2
    u_axis = np.arange(-u_max, u_max + 1)
    R = np.zeros((u_axis.size, n_angles))
    vals = np.real(img).ravel()
    for k, v in enumerate(v_axis):
        u = (ii * np.cos(v) + jj * np.sin(v) + u_max)
        R[:, k] = np.bincount(np.rint(u).astype(np.int64).ravel(),
                              weights=vals, minlength=u_axis.size)
    return R, v_axis, u_axis


def preprocess_tf(tfmap: np.ndarray, power: float = 1.0, prep: str = "none"):
    """Make |G(t,w)| fit for ridge detection.

    Two effects have to be handled.

    *Degenerate Radon directions.*  For v -> 0 the line of Eq. (12) runs along a
    whole time column, so the noise floor alone gives a huge R(u,v) and the
    ridge is missed.  `prep='white'` divides every frequency row by its temporal
    mean, which flattens that floor.

    *Comb-like behaviour along the slope.*  The Radon objective of a very sharp
    ridge oscillates with the slope, with a tooth spacing of about df/T_span
    (here ~1.7%).  Raising `power` concentrates the ridge further and makes the
    correct tooth dominate: with power = 2 the estimate saturates at a ~1.9%
    error at high GSNR, while power = 4 - the default of `radon_projection` -
    brings the median error down to ~1e-3.
    """
    mag = np.abs(np.asarray(tfmap))
    if power != 1.0:
        mag = mag ** power
    if prep == "white":
        # divide by the temporal mean of every frequency row: flattens the
        # noise floor and removes the degenerate v -> 0 Radon direction
        mag = mag / (mag.mean(axis=0, keepdims=True) + 1e-300)
    elif prep == "demean":
        mag = np.clip(mag - mag.mean(axis=0, keepdims=True), 0.0, None)
    elif prep == "none":
        pass
    else:
        raise ValueError(prep)
    return mag


def radon_projection(tf: GELCTResult | np.ndarray, f_axis, t_axis, df,
                     k_grid: np.ndarray, n_f0: int | None = None,
                     power: float = 4.0, prep: str = "white",
                     chunk: int = 128, stride: int = 1):
    """Radon transform written in slope/intercept coordinates.

    A straight line of the (t,w) plane is  w = f0 + k t  (in Hz, Hz/s).  With
    Eq. (12) this is exactly

        R(k,f0) = sum_t A(t, f0 + k t),     A = preprocessed |G|

    which is the same transform as R(u,v) - only parameterised by (slope,
    intercept) instead of (angle, offset).  That parameterisation is what makes
    the discrete version well behaved, because every line then integrates over
    the *same* number of samples.

    Returns (R_max(k), f0_best(k), full R).
    """
    A = preprocess_tf(tf if isinstance(tf, np.ndarray) else tf.G,
                      power=power, prep=prep)
    if stride > 1:
        A = A[::stride]
        t_axis = np.asarray(t_axis)[::stride]
    nt, nf = A.shape
    if n_f0 is None:
        n_f0 = nf
    f0g = np.linspace(f_axis[0], f_axis[-1], n_f0)
    rows = np.arange(nt)[None, :, None]          # (1, nt, 1)
    ti = t_axis
    R = np.empty((k_grid.size, n_f0))
    for a in range(0, k_grid.size, chunk):
        kk = k_grid[a:a + chunk]
        idx = (f0g[None, None, :] + kk[:, None, None] * ti[None, :, None]) / df
        np.clip(idx, 0, nf - 1, out=idx)          # (chunk, nt, n_f0)
        i0 = idx.astype(np.int32)
        wg = idx - i0
        i1 = np.minimum(i0 + 1, nf - 1)
        vals = A[rows, i0] * (1.0 - wg) + A[rows, i1] * wg
        R[a:a + chunk] = vals.sum(axis=1)
    imax = np.argmax(R, axis=1)
    return R[np.arange(k_grid.size), imax], f0g[imax], R


def _parabolic(x: np.ndarray, y: np.ndarray, i: int) -> float:
    """Sub-grid peak location from the three samples around index i."""
    if i <= 0 or i >= len(y) - 1:
        return float(x[i])
    d = y[i - 1] - 2.0 * y[i] + y[i + 1]
    if d == 0:
        return float(x[i])
    return float(x[i] + 0.5 * (y[i - 1] - y[i + 1]) / d * (x[1] - x[0]))


def default_k_max(n_samples: int, fs: float) -> float:
    """Largest meaningful chirp rate: sweeps the whole band within the record."""
    return fs ** 2 / (2.0 * n_samples)


def estimate_chirp_rate(tf: GELCTResult, k_max: float | None = None,
                        n_coarse: int = 641, n_medium: int = 401,
                        n_fine: int = 101, power: float = 4.0,
                        prep: str = "white", n_f0: int | None = None,
                        k_hint: float | None = None, hint_span: float = 0.25):
    """Eq. (13)-(14): Radon peak  ->  v_hat  ->  k_hat = -cot(v_hat) df/dt.

    Three cascaded grids (coarse / medium / fine) with parabolic interpolation
    of the peak are used, which is necessary because the angle grid of Eq. (14)
    alone limits the resolution to a few percent.

    Numerical caveat that motivates `k_hint`: the Radon objective of a *sharp*
    ridge is comb-like along k, with a tooth spacing of about df/T_span (here
    ~1.7%).  If the coarse grid step is of the same order, the search may lock
    onto a neighbouring tooth (a one-tooth error looks like a ~1.7% bias).  Pass
    a rough `k_hint` (e.g. from estimate_k_full_window) to restrict the coarse
    stage to +-`hint_span` around it and remove the ambiguity.
    """
    if k_max is None:
        k_max = default_k_max(tf.n_samples or 1024, tf.fs)  # full-band sweep rate
    # ----- stage 1: coarse ---------------------------------------------------
    if k_hint is not None:
        hw = max(abs(k_hint) * hint_span, 4.0 * tf.df / max(1e-12, tf.dt * tf.G.shape[0]))
        k_max = min(k_max, hw)
        centre = k_hint
    else:
        centre = 0.0
    kg = np.linspace(centre - k_max, centre + k_max, n_coarse)
    c, _, _ = radon_projection(tf, tf.f_axis, tf.t_axis, tf.df, kg,
                               n_f0=n_f0, power=power, prep=prep, stride=4)
    k1 = _parabolic(kg, c, int(np.argmax(c)))
    # ----- stage 2: medium ---------------------------------------------------
    span = 2.0 * (kg[1] - kg[0])
    kg2 = np.linspace(k1 - span, k1 + span, n_medium)
    c2, _, _ = radon_projection(tf, tf.f_axis, tf.t_axis, tf.df, kg2,
                                n_f0=n_f0, power=power, prep=prep, stride=2)
    k2 = _parabolic(kg2, c2, int(np.argmax(c2)))
    # ----- stage 3: fine -----------------------------------------------------
    kg3 = np.linspace(k2 - 0.02 * max(1.0, abs(k2)), k2 + 0.02 * max(1.0, abs(k2)),
                      n_fine)
    c3, f0_3, _ = radon_projection(tf, tf.f_axis, tf.t_axis, tf.df, kg3,
                                   n_f0=n_f0, power=power, prep=prep)
    i3 = int(np.argmax(c3))
    k_hat = _parabolic(kg3, c3, i3)

    # the angle of the detected line, for the paper's explicit formula
    slope_bins = k_hat * tf.dt / tf.df            # d(freq bin)/d(time bin)
    v_hat = np.arctan2(1.0, -slope_bins)          # -cot(v_hat) = slope_bins
    info = {"k_hat": k_hat, "v_hat": float(v_hat), "f0_radon": float(f0_3[i3]),
            "k_coarse": k1, "k_medium": k2, "curve": (kg3, c3),
            "k_grid_coarse": kg, "curve_coarse": c}
    return k_hat, info


# ############################################################################
# 5.  Generalized Fourier transform and initial-frequency estimation
# ############################################################################
def generalized_fourier_transform(x: np.ndarray, fs: float,
                                  nfft: int | None = None):
    """Eq. (16):  GF(w) = int f[x(t)] exp(-j w t) dt.  ->  (GF, freqs[Hz])."""
    x = np.asarray(x)
    nfft = nfft or x.size
    gf = np.fft.fftshift(np.fft.fft(nonlinear_transform(x), nfft))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    return gf, freqs


def estimate_initial_frequency(r: np.ndarray, k_hat: float, fs: float,
                               nfft: int | None = None):
    """Eq. (15) + (20): de-chirp by the estimated rate, take the GFT and read
    the position of the spectral maximum (with sub-bin interpolation)."""
    r = np.asarray(r)
    t = np.arange(r.size) / fs
    r1 = r * np.exp(-1j * np.pi * k_hat * t ** 2)             # Eq. (15)
    gf, freqs = generalized_fourier_transform(r1, fs, nfft)   # Eq. (16)
    mag = np.abs(gf)
    j = int(np.argmax(mag))
    f0 = _parabolic(freqs, mag, j) if 0 < j < mag.size - 1 else float(freqs[j])
    return float(f0), {"GF": gf, "freqs": freqs, "peak_index": j}


# ############################################################################
# 6.  Full estimator  (Algorithm 1 + Algorithm 2)
# ############################################################################
def estimate_lfm_params(r: np.ndarray, fs: float, ts: float,
                        cfg: GELCTConfig | None = None,
                        k_max: float | None = None,
                        nfft_gft: int | None = None,
                        auto_hint: bool = False, **kw):
    """Joint estimation of (chirp rate, initial frequency) of the LFM
    interference buried in S-alpha-S noise.

        Algorithm 1 :  f[r] -> GELCT -> Radon -> k_hat
        Algorithm 2 :  r1 = r exp(-j pi k_hat t^2) -> GFT -> f0_hat

    auto_hint=True first runs the full-window variant (Sec. 3.4 of the README)
    and uses its result to restrict the Radon search, which removes the
    comb-tooth ambiguity of the discrete short-time Radon.
    """
    tf = gelct(r, fs, ts, cfg)
    if auto_hint and "k_hint" not in kw:
        kw["k_hint"] = estimate_k_full_window(r, fs, k_max=k_max)
    k_hat, rad = estimate_chirp_rate(tf, k_max=k_max, **kw)
    f0_hat, gft = estimate_initial_frequency(r, k_hat, fs, nfft_gft)
    return {"k_hat": k_hat, "f0_hat": f0_hat, "tf": tf,
            "radon": rad, "gft": gft}


def coherent_dechirp_scan(r: np.ndarray, fs: float, k_grid: np.ndarray,
                          chunk: int = 1024, metric: str = "peak_ratio"):
    """Radon transform of the GELCT plane in the limit of a full-length window.

    Because the nonlinear gain of Eq. (8) depends on |r| only, f[.] commutes
    with the unit-modulus de-chirping factor:

        f[ r(t) exp(-j pi k t^2) ] = f[r(t)] exp(-j pi k t^2),

    so a full-length-window GELCT followed by the Radon transform collapses to a
    one-dimensional scan over k of the peak of |FFT(f[r] exp(-j pi k t^2))|.
    This variant integrates coherently over all N samples (no window loss) and
    is therefore markedly more accurate than the short-time plane; it is used
    for the refined curves and to cross-check the Radon estimate.
    """
    z0 = nonlinear_transform(r)
    t2 = (np.arange(r.size) / fs) ** 2
    score = np.empty(k_grid.size)
    for a in range(0, k_grid.size, chunk):
        sub = k_grid[a:a + chunk]
        Z = z0[:, None] * np.exp(-1j * np.pi * sub[None, :] * t2[:, None])
        S = np.abs(np.fft.fft(Z, axis=0))
        if metric == "peak_ratio":
            score[a:a + chunk] = S.max(axis=0) ** 2 / (np.sum(S ** 2, axis=0) + 1e-300)
        else:
            score[a:a + chunk] = S.max(axis=0)
    return score


def refine_k_coherent(r: np.ndarray, fs: float, k_guess: float,
                      span: float = 0.05, n_coarse: int = 801,
                      n_fine: int = 401):
    """Polish k by a local coherent de-chirp scan around k_guess."""
    kg = np.linspace(k_guess * (1 - span), k_guess * (1 + span), n_coarse)
    c = coherent_dechirp_scan(r, fs, kg)
    k1 = _parabolic(kg, c, int(np.argmax(c)))
    kg2 = np.linspace(k1 - (kg[1] - kg[0]), k1 + (kg[1] - kg[0]), n_fine)
    c2 = coherent_dechirp_scan(r, fs, kg2)
    return _parabolic(kg2, c2, int(np.argmax(c2)))


def estimate_k_full_window(r: np.ndarray, fs: float, k_max: float | None = None,
                           n_coarse: int = 1601, n_fine: int = 401):
    """Chirp-rate estimate from the *full-length window* GELCT + Radon.

    Setting win_len = N in Eq. (7) leaves a single time frame, and the Radon
    transform of the resulting (theta, w) family is exactly the one-dimensional
    scan of coherent_dechirp_scan.  All N samples are then integrated with a
    constant phase weight, so this variant - unlike the short-time plane - does
    not lose the window length in processing gain.
    """
    if k_max is None:
        k_max = default_k_max(r.size, fs)
    kg = np.linspace(-k_max, k_max, n_coarse)
    c = coherent_dechirp_scan(r, fs, kg)
    k1 = _parabolic(kg, c, int(np.argmax(c)))
    span = 2.0 * (kg[1] - kg[0])
    kg2 = np.linspace(k1 - span, k1 + span, n_fine)
    c2 = coherent_dechirp_scan(r, fs, kg2)
    k2 = _parabolic(kg2, c2, int(np.argmax(c2)))
    kg3 = np.linspace(k2 - 0.25 * span, k2 + 0.25 * span, 201)
    c3 = coherent_dechirp_scan(r, fs, kg3)
    return _parabolic(kg3, c3, int(np.argmax(c3)))

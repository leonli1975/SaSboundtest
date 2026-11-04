# -*- coding: utf-8 -*-
"""
Alpha-stable (SalphaS) noise generation, distribution statistics and CRLBs.

Reference
---------
Liu, M., Han, Y., Chen, Y., Song, H., Yang, Z., Gong, F. (2021).
"Modulation Parameter Estimation of LFM Interference for Direct Sequence
 Spread Spectrum Communication System in Alpha-Stable Noise",
IEEE Systems Journal, 15(1), 881-892.   DOI: 10.1109/JSYST.2020.2991078

Eq. numbering below follows that paper.
"""
from __future__ import annotations

import numpy as np

E = np.e          # Euler's number, used by the nonlinear transform f[.]


# ----------------------------------------------------------------------------
# S-alpha-S random variables  (Chambers-Mallows-Stuck, beta = 0)
# ----------------------------------------------------------------------------
def sas_rvs(alpha: float, gamma: float, size, rng: np.random.Generator):
    """Draw i.i.d. samples of S(alpha, beta=0, gamma, delta=0).

    Uses the Chambers-Mallows-Stuck (CMS) direct method.  For alpha == 1 the
    distribution reduces to a Cauchy law, for alpha == 2 to a Gaussian law with
    variance 2*gamma (this matches the convention used in the paper).
    """
    if not (0.0 < alpha <= 2.0):
        raise ValueError("alpha must lie in (0, 2]")
    if alpha == 2.0:
        # closed form: N(0, 2*gamma) -- avoids 0/0 in the generic expression
        return np.sqrt(2.0 * gamma) * rng.standard_normal(size)

    v = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size)      # V  ~ U(-pi/2, pi/2)
    w = rng.exponential(1.0, size)                        # W  ~ Exp(1)

    if abs(alpha - 1.0) < 1e-12:
        # Cauchy: S(1, 0, gamma, 0) -> density gamma / (pi * (gamma^2 + x^2))
        x = np.tan(v)
    else:
        x = (np.sin(alpha * v) / np.power(np.cos(v), 1.0 / alpha)
             * np.power(np.cos((1.0 - alpha) * v) / w, (1.0 - alpha) / alpha))
    return np.power(gamma, 1.0 / alpha) * x


def complex_sas_noise(alpha: float, gamma: float, n: int, rng: np.random.Generator):
    """Complex isotropic-like S-alpha-S noise.

    Following Eq. (35)-(40) the real and the imaginary part are i.i.d.
    S(alpha, 0, gamma, 0), therefore each quadrature carries the same
    dispersion gamma.
    """
    re = sas_rvs(alpha, gamma, n, rng)
    im = sas_rvs(alpha, gamma, n, rng)
    return re + 1j * im


def gsnr_to_gamma(gsnr_db: float, signal_power: float) -> float:
    """Eq. (6):  GSNR = 10*log10(sigma_s^2 / gamma)  ->  gamma."""
    return signal_power / (10.0 ** (gsnr_db / 10.0))


# ----------------------------------------------------------------------------
# kappa(alpha) = int (f'(x))^2 / f(x) dx     (appears in the Fisher information)
# ----------------------------------------------------------------------------
def kappa_alpha(alpha: float, half_width: float = 400.0, n_grid: int = 1 << 21):
    """Numerically integrate kappa(alpha) for S(alpha, 0, gamma=1, 0).

    The density is obtained by inverse Fourier transforming the characteristic
    function exp(-|t|^alpha) (Eq. (3) with beta = delta = 0, gamma = 1), the
    logarithmic derivative is taken by finite differences and kappa is the
    trapezoidal integral of f * (d log f / dx)^2.

    Reference values (Eq. (46) and (49) of the paper): kappa(1) = 0.5 and
    kappa(2) = 0.5.
    """
    if alpha == 2.0:
        return 0.5
    if alpha == 1.0:
        return 0.5

    dx = 2.0 * half_width / n_grid
    x = (np.arange(n_grid) - n_grid // 2) * dx            # symmetric grid
    dt = 2.0 * np.pi / (n_grid * dx)                      # dual grid spacing
    t = (np.arange(n_grid) - n_grid // 2) * dt
    cf = np.exp(-np.power(np.abs(t), alpha))              # phi(t)
    f = np.fft.fftshift(np.real(np.fft.ifft(np.fft.ifftshift(cf)))) / dx
    f = np.clip(f, 1e-300, None)

    dlogf = np.gradient(np.log(f), dx)
    integrand = f * dlogf ** 2
    return float(np.trapezoid(integrand, x))


# ----------------------------------------------------------------------------
# Cramer-Rao lower bounds, Eq. (44), (45)
# ----------------------------------------------------------------------------
def crlb_initial_frequency(n_samples: int, ts: float, kappa: float,
                           gsnr_linear: float) -> float:
    """Eq. (44): CRLB(f0) = 6 / (4*kappa*GSNR*pi^2*Ts^2*(2N^3+3N^2+N))."""
    n = n_samples
    denom = (4.0 * kappa * gsnr_linear * np.pi ** 2 * ts ** 2
             * (2.0 * n ** 3 + 3.0 * n ** 2 + n))
    return 6.0 / denom


def crlb_chirp_rate(n_samples: int, ts: float, kappa: float,
                    gsnr_linear: float) -> float:
    """Eq. (45): CRLB(k) = 30 / (kappa*GSNR*pi^2*Ts^4*(6N^5+15N^4+10N^3))."""
    n = n_samples
    denom = (kappa * gsnr_linear * np.pi ** 2 * ts ** 4
             * (6.0 * n ** 5 + 15.0 * n ** 4 + 10.0 * n ** 3))
    return 30.0 / denom

"""
bcc_pricing.py
==============
European option pricing via the Lewis (2001) Fourier transform.

Implements characteristic functions for:
  - Black-Scholes-Merton (BSM)
  - Merton (1976) jump-diffusion (M76)
  - Heston (1993) stochastic volatility (H93)
  - Bakshi-Cao-Chen (1997) unified model (BCC)

All models are priced through a single Lewis transform function,
demonstrating the generality of the characteristic function approach.

Reference: Hilpisch (2015), Derivatives Analytics with Python,
           Chapters 8-9.
"""

import numpy as np
from scipy.integrate import quad

def bsm_call_value(S0, K, T, r, sigma):
    """
    Black-Scholes-Merton European call price (closed form).

    Parameters
    ----------
    S0 : float
        Current stock price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Constant volatility.

    Returns
    -------
    float
        European call option price.
    """
    from scipy.stats import norm
    if T <= 0:
        return max(S0 - K, 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (
        sigma * np.sqrt(T)
    )
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def lewis_call_value(S0, K, T, r, char_func, *args):
    """
    European call price via the Lewis (2001) Fourier transform.

    Works for any model with a known characteristic function.
    The same function prices BSM, Merton, Heston, and BCC —
    only the characteristic function argument changes.

    Parameters
    ----------
    S0 : float
        Current stock price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    char_func : callable
        Characteristic function of the log-return.
        Signature: char_func(u, T, r, *args) -> complex
    *args :
        Additional parameters passed to char_func.

    Returns
    -------
    float
        European call option price.
    """
    if T <= 0:
        return max(S0 - K, 0.0)

    k = np.log(K / S0)   # log moneyness

    def integrand(u):
        cf = char_func(u - 0.5j, T, r, *args)
        numerator = np.exp(-1j * u * k) * cf
        denominator = u ** 2 + 0.25
        return (numerator / denominator).real

    integral, _ = quad(integrand, 0, 100, limit=500,
                       epsabs=1e-8, epsrel=1e-8)

    call = S0 - (np.sqrt(S0 * K) * np.exp(-r * T) / np.pi) * integral
    return max(call, 0.0)


def char_func_bsm(u, T, r, sigma):
    """
    Characteristic function of the log-return under BSM.

    Parameters
    ----------
    u : complex
        Frequency variable.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    sigma : float
        Constant volatility.

    Returns
    -------
    complex
        Characteristic function value at u.
    """
    return np.exp(
        (1j * u * (r - 0.5 * sigma ** 2) * T)
        - (0.5 * sigma ** 2 * u ** 2 * T)
    )

def char_func_m76(u, T, r, sigma, lamb, mu_j, delta):
    """
    Characteristic function of the log-return under Merton (1976)
    jump-diffusion.

    Parameters
    ----------
    u : complex
        Frequency variable.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    sigma : float
        Diffusion volatility.
    lamb : float
        Jump intensity (jumps per year).
    mu_j : float
        Mean log jump size.
    delta : float
        Jump size volatility.

    Returns
    -------
    complex
        Characteristic function value at u.
    """
    # Jump compensation term — keeps model risk-neutral
    r_j = lamb * (np.exp(mu_j + 0.5 * delta ** 2) - 1)

    # Characteristic function of jump size distribution
    omega = np.exp(1j * u * mu_j - 0.5 * delta ** 2 * u ** 2) - 1

    return np.exp(
        (1j * u * (r - r_j - 0.5 * sigma ** 2) * T)
        - (0.5 * sigma ** 2 * u ** 2 * T)
        + (lamb * omega * T)
    )


def char_func_h93(u, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    """
    Characteristic function of the log-return under Heston (1993)
    stochastic volatility model.

    Parameters
    ----------
    u : complex
        Frequency variable.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    kappa_v : float
        Speed of mean reversion for variance.
    theta_v : float
        Long-run variance.
    sigma_v : float
        Volatility of variance (vol of vol).
    rho : float
        Correlation between stock and variance shocks.
    v0 : float
        Initial variance.

    Returns
    -------
    complex
        Characteristic function value at u.
    """
    # Complex drift adjustment
    alpha = -0.5 * u ** 2 - 0.5 * 1j * u

    # Adjusted mean reversion incorporating correlation
    beta = kappa_v - rho * sigma_v * 1j * u

    # Combined term
    gamma = 0.5 * sigma_v ** 2

    # Discriminant
    d = np.sqrt(beta ** 2 - 4 * alpha * gamma)

    # Ratio term
    g = (beta - d) / (beta + d)

    # Time-dependent components
    C = (kappa_v * theta_v / sigma_v ** 2) * (
        (beta - d) * T
        - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    )

    D = ((beta - d) / sigma_v ** 2) * (
        (1 - np.exp(-d * T))
        / (1 - g * np.exp(-d * T))
    )

    return np.exp(C + D * v0 + 1j * u * r * T)

def char_func_bcc(u, T, r, kappa_v, theta_v, sigma_v,
                  rho, v0, lamb, mu_j, delta):
    """
    Characteristic function of the log-return under the
    Bakshi-Cao-Chen (1997) unified model.

    Combines Heston (1993) stochastic volatility with
    Merton (1976) jump-diffusion. The CIR interest rate
    component is handled separately via the equivalent
    flat rate r passed as input.

    Parameters
    ----------
    u : complex
        Frequency variable.
    T : float
        Time to maturity.
    r : float
        Equivalent flat rate from CIR calibration.
    kappa_v : float
        Speed of mean reversion for variance.
    theta_v : float
        Long-run variance.
    sigma_v : float
        Volatility of variance.
    rho : float
        Correlation between stock and variance shocks.
    v0 : float
        Initial variance.
    lamb : float
        Jump intensity (jumps per year).
    mu_j : float
        Mean log jump size.
    delta : float
        Jump size volatility.

    Returns
    -------
    complex
        Characteristic function value at u.
    """
    # Jump compensation — keeps model risk-neutral
    r_j = lamb * (np.exp(mu_j + 0.5 * delta ** 2) - 1)

    # Heston component — same as H93 but with adjusted drift
    alpha = -0.5 * u ** 2 - 0.5 * 1j * u
    beta  = kappa_v - rho * sigma_v * 1j * u
    gamma = 0.5 * sigma_v ** 2

    d = np.sqrt(beta ** 2 - 4 * alpha * gamma)
    g = (beta - d) / (beta + d)

    C = (kappa_v * theta_v / sigma_v ** 2) * (
        (beta - d) * T
        - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    )

    D = ((beta - d) / sigma_v ** 2) * (
        (1 - np.exp(-d * T))
        / (1 - g * np.exp(-d * T))
    )

    # Jump component
    omega = np.exp(1j * u * mu_j - 0.5 * delta ** 2 * u ** 2) - 1

    # Combined BCC characteristic function
    return np.exp(
        C + D * v0
        + 1j * u * (r - r_j) * T
        + lamb * omega * T
    )




"""
cir_model.py
============
Cox-Ingersoll-Ross (1985) short-rate model.

Functions for zero-coupon bond pricing, yield calculation,
forward rate computation, and calibration to a market
yield curve.

Reference: Hilpisch (2015), Derivatives Analytics with Python,
           Chapter 9.
"""

import numpy as np
from scipy.optimize import fmin

def cir_zcb_value(r0, kappa_r, theta_r, sigma_r, T, t=0.0):
    """
    CIR zero-coupon bond price.

    Parameters
    ----------
    r0 : float
        Current short rate.
    kappa_r : float
        Speed of mean reversion.
    theta_r : float
        Long-run mean rate.
    sigma_r : float
        Volatility of the short rate.
    T : float
        Bond maturity in years.
    t : float, optional
        Current time (default 0).

    Returns
    -------
    float
        Zero-coupon bond price B(t, T).
    """
    tau = T - t
    if tau <= 0:
        return 1.0

    gamma = np.sqrt(kappa_r ** 2 + 2 * sigma_r ** 2)
    denom = ((kappa_r + gamma) * (np.exp(gamma * tau) - 1)
             + 2 * gamma)

    A = ((2 * gamma * np.exp((kappa_r + gamma) * tau / 2))
         / denom) ** (2 * kappa_r * theta_r / sigma_r ** 2)

    C = (2 * (np.exp(gamma * tau) - 1)) / denom

    return A * np.exp(-C * r0)
  

def cir_zcb_yield(r0, kappa_r, theta_r, sigma_r, T, t=0.0):
    """
    Continuously compounded yield implied by the CIR bond price.

    Returns
    -------
    float
        Yield y(t, T) = -ln(B(t,T)) / (T - t).
    """
    tau = T - t
    if tau <= 0:
        return r0
    price = cir_zcb_value(r0, kappa_r, theta_r, sigma_r, T, t)
    return -np.log(price) / tau


def cir_forward_rate(r0, kappa_r, theta_r, sigma_r, T):
    """
    Instantaneous forward rate implied by the CIR model.

    Computed as the negative derivative of ln(B(0,T)) with
    respect to T, approximated numerically.

    Returns
    -------
    float
        Forward rate f(0, T).
    """
    dt = 0.0001
    B1 = cir_zcb_value(r0, kappa_r, theta_r, sigma_r, T)
    B2 = cir_zcb_value(r0, kappa_r, theta_r, sigma_r, T + dt)
    return -np.log(B2 / B1) / dt

def cir_calibrate(market_tenors, market_rates, r0,
                  x0=None, maxiter=500):
    """
    Calibrate CIR parameters to a market yield curve.

    Parameters
    ----------
    market_tenors : array-like
        Maturities in years.
    market_rates : array-like
        Market zero-coupon rates (as decimals, e.g. 0.07).
    r0 : float
        Starting short rate (overnight rate).
    x0 : array-like, optional
        Initial guess [kappa_r, theta_r, sigma_r].
    maxiter : int
        Maximum optimisation iterations.

    Returns
    -------
    dict
        Calibrated parameters and RMSE.
    """
    market_tenors = np.asarray(market_tenors)
    market_rates = np.asarray(market_rates)

    if x0 is None:
        x0 = [0.3, market_rates.mean(), 0.1]

    def error_function(params):
        kappa_r, theta_r, sigma_r = params
        # Enforce positive parameters
        if kappa_r <= 0 or theta_r <= 0 or sigma_r <= 0:
            return 1e10
        model_rates = np.array([
            cir_zcb_yield(r0, kappa_r, theta_r, sigma_r, T)
            for T in market_tenors
        ])
        return np.sum((market_rates - model_rates) ** 2)

    result = fmin(error_function, x0,
                  maxiter=maxiter, disp=False)

    kappa_r, theta_r, sigma_r = result

    # Compute final RMSE
    model_rates = np.array([
        cir_zcb_yield(r0, kappa_r, theta_r, sigma_r, T)
        for T in market_tenors
    ])
    rmse = np.sqrt(np.mean((market_rates - model_rates) ** 2))

    return {
        'r0': r0,
        'kappa_r': float(kappa_r),
        'theta_r': float(theta_r),
        'sigma_r': float(sigma_r),
        'rmse': float(rmse),
    }

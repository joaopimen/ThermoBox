import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from activity_coeff import calculate_activity_coefficients as lngamma_calc

# --------------------------------------------------
# Data: Antoine coefficients
# --------------------------------------------------

_db = pd.read_csv("antoine_coeff.csv")
if "comp" in _db.columns:
    _db = _db.set_index("comp")


def _antoine_params(comp_list):
    """Return vectors A, B, C for the given component name list."""
    rows = _db.loc[comp_list]
    A = rows["A"].to_numpy(dtype=float)
    B = rows["B"].to_numpy(dtype=float)
    C = rows["C"].to_numpy(dtype=float)
    return A, B, C


def _psat_mmHg(A, B, C, T_C):
    """Antoine equation -> saturation pressure in mmHg."""
    T_C = float(T_C)
    return 10.0 ** (A - B / (T_C + C))


# --------------------------------------------------
# Dew-point: T known
# --------------------------------------------------

def dew_point_T_known(comp, n, T_K, params, method="unifac-vle"):
    """
    Dew-point calculation at known temperature.

    Parameters
    ----------
    comp : list[str]
        Component names in the same order as n.
    n : array-like
        Overall moles of each component (vapor overall composition).
    T_K : float
        Temperature in Kelvin.
    params : dict
        Parameter dictionary (UNIFAC, etc.).
    method : str
        Activity-coefficient method string for lngamma_calc.

    Returns
    -------
    x : ndarray
        Liquid-phase equilibrium mole fractions.
    Psat : ndarray
        Saturation pressures (mmHg) at T.
    P : float
        System pressure (mmHg).
    gamma : ndarray
        Activity coefficients in the liquid phase.
    """
    n = np.asarray(n, dtype=float)
    z = n / n.sum()      # overall vapor composition
    y = z.copy()
    NC = len(comp)

    T_C = T_K - 273.15
    A, B, C = _antoine_params(comp)
    Psat = _psat_mmHg(A, B, C, T_C)

    # Ideal initial pressure estimate (gamma = 1)
    P0 = 1.0 / np.sum(y / Psat)

    def objective(P):
        P = float(P)
        x = y * P / Psat
        x /= x.sum()
        lngamma = lngamma_calc(x, n, T_K, NC, method, comp, params)
        gamma = np.maximum(np.exp(lngamma), 1e-10)
        Ki = gamma * Psat / P
        return np.sum(y / Ki) - 1.0

    P = float(fsolve(objective, P0)[0])

    x = y * P / Psat
    x /= x.sum()
    lngamma = lngamma_calc(x, n, T_K, NC, method, comp, params)
    gamma = np.maximum(np.exp(lngamma), 1e-10)

    return x, Psat, P, gamma


# --------------------------------------------------
# Dew-point: P known
# --------------------------------------------------

def dew_point_P_known(comp, n, P, params, method="unifac-vle"):
    """
    Dew-point calculation at known pressure.

    Parameters
    ----------
    comp : list[str]
        Component names.
    n : array-like
        Overall moles (vapor overall composition).
    P : float
        System pressure (mmHg).
    params : dict
        Parameter dictionary.
    method : str
        Activity-coefficient method string.

    Returns
    -------
    x : ndarray
        Liquid-phase equilibrium mole fractions.
    T_K : float
        Dew-point temperature (K).
    gamma : ndarray
        Activity coefficients in the liquid phase.
    """
    n = np.asarray(n, dtype=float)
    z = n / n.sum()
    y = z.copy()
    NC = len(comp)
    P = float(P)

    A, B, C = _antoine_params(comp)

    def objective(T_C):
        T_C = float(T_C)
        Psat = _psat_mmHg(A, B, C, T_C)
        T_K = T_C + 273.15

        x = y * P / Psat
        x /= x.sum()

        lngamma = lngamma_calc(x, n, T_K, NC, method, comp, params)
        gamma = np.maximum(np.exp(lngamma), 1e-10)
        Ki = gamma * Psat / P
        return np.sum(y / Ki) - 1.0

    # crude initial guess for T (°C)
    T_C0 = 60.0
    T_C = float(fsolve(objective, T_C0)[0])
    Psat = _psat_mmHg(A, B, C, T_C)
    T_K = T_C + 273.15

    x = y * P / Psat
    x /= x.sum()
    lngamma = lngamma_calc(x, n, T_K, NC, method, comp, params)
    gamma = np.maximum(np.exp(lngamma), 1e-10)

    return x, T_K, gamma


# --------------------------------------------------
# Bubble-point: T known
# --------------------------------------------------

def bubble_point_T_known(comp, n, T_K, params, method="unifac-vle"):
    """
    Bubble-point calculation at known temperature.

    Parameters
    ----------
    comp : list[str]
        Component names.
    n : array-like
        Overall moles of each component (liquid overall composition).
    T_K : float
        Temperature in Kelvin.
    params : dict
        Parameter dictionary.
    method : str
        Activity-coefficient method string.

    Returns
    -------
    y : ndarray
        Vapor-phase equilibrium mole fractions.
    Psat : ndarray
        Saturation pressures (mmHg) at T.
    P : float
        System pressure (mmHg).
    gamma : ndarray
        Activity coefficients in the liquid phase.
    """
    n = np.asarray(n, dtype=float)
    z = n / n.sum()      # overall liquid composition
    x = z.copy()
    NC = len(comp)

    T_C = T_K - 273.15
    A, B, C = _antoine_params(comp)
    Psat = _psat_mmHg(A, B, C, T_C)

    lngamma = lngamma_calc(x, n, T_K, NC, method, comp, params)
    gamma = np.maximum(np.exp(lngamma), 1e-10)

    P = float(np.sum(x * gamma * Psat))
    y = x * gamma * Psat / P
    y /= y.sum()

    return y, Psat, P, gamma


# --------------------------------------------------
# Bubble-point: P known
# --------------------------------------------------

def bubble_point_P_known(comp, n, P, params, method="unifac-vle"):
    """
    Bubble-point calculation at known pressure.

    Parameters
    ----------
    comp : list[str]
        Component names.
    n : array-like
        Overall moles (liquid overall composition).
    P : float
        System pressure (mmHg).
    params : dict
        Parameter dictionary.
    method : str
        Activity-coefficient method string.

    Returns
    -------
    y : ndarray
        Vapor-phase equilibrium mole fractions.
    T_K : float
        Bubble-point temperature (K).
    gamma : ndarray
        Activity coefficients in the liquid phase.
    """
    n = np.asarray(n, dtype=float)
    z = n / n.sum()
    x = z.copy()
    NC = len(comp)
    P = float(P)

    A, B, C = _antoine_params(comp)

    def objective(T_C):
        T_C = float(T_C)
        Psat = _psat_mmHg(A, B, C, T_C)
        T_K = T_C + 273.15
        lngamma = lngamma_calc(x, n, T_K, NC, method, comp, params)
        gamma = np.maximum(np.exp(lngamma), 1e-10)
        return np.sum(x * gamma * Psat) - P

    # crude initial guess for T (°C)
    T_C0 = 60.0
    T_C = float(fsolve(objective, T_C0)[0])
    T_K = T_C + 273.15
    Psat = _psat_mmHg(A, B, C, T_C)
    lngamma = lngamma_calc(x, n, T_K, NC, method, comp, params)
    gamma = np.maximum(np.exp(lngamma), 1e-10)
    y = x * gamma * Psat / P
    y /= y.sum()

    return y, T_K, gamma

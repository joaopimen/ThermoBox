import warnings
warnings.filterwarnings("ignore")

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


def _antoine_params(comp_list, T_C):

    """Return vectors A, B, C for the given component name list."""
    rows = _db.loc[comp_list]

    T_min = rows["Tmin"].to_numpy(dtype=float)
    T_max = rows["Tmax"].to_numpy(dtype=float)

    if T_min.any() > T_C:
           
        rows = rows.drop_duplicates(subset='fq', keep='first')

        A = rows["A"].to_numpy(dtype=float)
        B = rows["B"].to_numpy(dtype=float)
        C = rows["C"].to_numpy(dtype=float)

    elif T_max.any() < T_C:

        rows = rows.drop_duplicates(subset='fq', keep='last')

        A = rows["A"].to_numpy(dtype=float)
        B = rows["B"].to_numpy(dtype=float)
        C = rows["C"].to_numpy(dtype=float)

    else:

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

def dew_point_T_known(comp, n, T_K, params, method="unifac"):
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
        Saturation pressures (mmHg) at T (°C).
    P_bar : float
        System pressure (bar).
    gamma : ndarray
        Activity coefficients in the liquid phase.
    """
    NC = len(comp)
    n = np.asarray(n, dtype=float)
    y = n / n.sum()      # overall vapor composition
    
    T_C = T_K - 273.15
    A, B, C = _antoine_params(comp, T_C)
    Psat = _psat_mmHg(A, B, C, T_C)

    def objective(Z):
        P, *x = Z

        lngamma = lngamma_calc(x, n, T_K, NC, method, comp, params)
        gamma = np.maximum(np.exp(lngamma), 1e-10)

        Ki = gamma * Psat / P

        eqP = 1 - np.sum(y / Ki)
        eqx = x * Ki - y
        
        return np.concatenate((eqP, eqx), axis=None)

    # Ideal initial pressure estimate (gamma = 1)
    P0 = 1.0 / np.sum(y / Psat)
    x0 = y * P0 / Psat

    P_mmHg, *x = fsolve(objective, np.concatenate((P0, x0), axis=None))

    lngamma = lngamma_calc(x, n, T_K, NC, method, comp, params)
    gamma = np.maximum(np.exp(lngamma), 1e-10)

    P_bar = P_mmHg / 750.06157593

    return x, P_bar, Psat, gamma


# --------------------------------------------------
# Dew-point: P known
# --------------------------------------------------

def dew_point_P_known(comp, n, P_bar, params, method="unifac"):
    """
    Dew-point calculation at known pressure.

    Parameters
    ----------
    comp : list[str]
        Component names.
    n : array-like
        Overall moles (vapor overall composition).
    P_bar : float
        System pressure (bar).
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
    NC = len(comp)
    n = np.asarray(n, dtype=float)
    y = n / n.sum()
    P_mmHg = P_bar * 750.06157593

    def objective(Z):
        T_C, *x = Z

        A, B, C = _antoine_params(comp, T_C)
        Psat = _psat_mmHg(A, B, C, T_C)
        T_K = T_C + 273.15

        lngamma = lngamma_calc(x, n, T_K, NC, method, comp, params)
        gamma = np.maximum(np.exp(lngamma), 1e-10)

        Ki = gamma * Psat / P_mmHg

        eqT = 1 - np.sum(y / Ki)
        eqx = x * Ki - y
        
        return np.concatenate((eqT, eqx), axis=None)

    # crude initial guess for x (liquid overall composition) and T (dew-point temperature [°C])
    x0 = y    
    T_C0 = 60.0
    T_C, *x = fsolve(objective, np.concatenate((T_C0, x0), axis=None))
    
    A, B, C = _antoine_params(comp, T_C)
    Psat = _psat_mmHg(A, B, C, T_C)

    T_K = T_C + 273.15
    lngamma = lngamma_calc(x, n, T_K, NC, method, comp, params)
    gamma = np.maximum(np.exp(lngamma), 1e-10)

    return x, T_K, Psat, gamma


# --------------------------------------------------
# Bubble-point: T known
# --------------------------------------------------

def bubble_point_T_known(comp, n, T_K, params, method="unifac"):
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
    P_bar : float
        System pressure (bar).
    gamma : ndarray
        Activity coefficients in the liquid phase.
    """
    NC = len(comp)
    n = np.asarray(n, dtype=float)
    x = n / n.sum()      # overall liquid composition
    
    T_C = T_K - 273.15
    A, B, C = _antoine_params(comp, T_C)
    Psat = _psat_mmHg(A, B, C, T_C)

    lngamma = lngamma_calc(x, n, T_K, NC, method, comp, params)
    gamma = np.maximum(np.exp(lngamma), 1e-10)

    P_mmHg = float(np.sum(x * gamma * Psat))
    y = x * gamma * Psat / P_mmHg

    P_bar = P_mmHg / 750.06157593

    return y, P_bar, Psat, gamma


# --------------------------------------------------
# Bubble-point: P known
# --------------------------------------------------

def bubble_point_P_known(comp, n, P_bar, params, method="unifac"):
    """
    Bubble-point calculation at known pressure.

    Parameters
    ----------
    comp : list[str]
        Component names.
    n : array-like
        Overall moles (liquid overall composition).
    P_bar : float
        System pressure (bar).
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
    NC = len(comp)
    n = np.asarray(n, dtype=float)
    x = n / n.sum()
    P_mmHg = P_bar * 750.06157593

    def objective(T_C):
        A, B, C = _antoine_params(comp, T_C)
        Psat = _psat_mmHg(A, B, C, T_C)
        T_K = T_C + 273.15
        lngamma = lngamma_calc(x, n, T_K, NC, method, comp, params)
        gamma = np.maximum(np.exp(lngamma), 1e-10)
        return np.sum(x * gamma * Psat) - P_mmHg

    # crude initial guess for T (bubble-point temperature [°C])
    T_C0 = 60.0
    T_C = fsolve(objective, T_C0)[0]
    T_K = T_C + 273.15
    A, B, C = _antoine_params(comp, T_C)
    Psat = _psat_mmHg(A, B, C, T_C)
    lngamma = lngamma_calc(x, n, T_K, NC, method, comp, params)
    gamma = np.maximum(np.exp(lngamma), 1e-10)
    y = x * gamma * Psat / P_mmHg

    return y, T_K, Psat, gamma


# from get_unifaclle_parameters import get_unifac_parameters
# comp = ['ETHANOL', 'WATER']
# params = get_unifac_parameters(comp)

# n1 = [1,1]
# P_bar1 = 1.5
# y, T_K1, gamma1 = bubble_point_P_known(comp, n1, P_bar1, params, method="unifac")
# print(y)
# print(T_K1)
# print(gamma1)

# n2 = [0.64164426, 0.35835574]
# T_K2 = 364.96016788502607
# x, P_bar2, gamma2 = dew_point_T_known(comp, n2, T_K2, params, method="unifac")
# print(x)
# print(P_bar2)
# print(gamma2)

# n1 = [1,1]
# P_bar1 = 1.5
# x, T_K1, gamma1 = dew_point_P_known(comp, n1, P_bar1, params, method="unifac")
# print(x)
# print(T_K1)
# print(gamma1)

# n2 = [0.22194701058174354, 0.7780529894042226]
# T_K2 = 368.2751716694585
# y, P_bar2, gamma2 = bubble_point_T_known(comp, n2, T_K2, params, method="unifac")
# print(y)
# print(P_bar2)
# print(gamma2)
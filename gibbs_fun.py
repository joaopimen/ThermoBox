from scipy.optimize import minimize
import numpy as np
import unifac
import unifac_parameters as up
from pyswarm import pso
# pip install --upgrade pyswarm

from activity_coeff import calculate_activity_coefficients as lngamma_calc

def gibbs_min(N, Z, T, R, NC, comp, method, params):
    NP = 2                  # Number of phases (currently only available for 2)
    # Allocates the initial guesses to each phase
    N_I  = np.array(N[:NC])
    # N_II = np.array(N[NC:])
    N_II = Z - N_I
    # Convert total moles in the feed to mole fractions
    z_mole_frac = Z / np.sum(Z)

    # Molar fractions
    xI = N_I / np.clip(np.nansum(N_I), 1e-12, None)
    xII = N_II / np.clip(np.nansum(N_II), 1e-12, None)

    # Avoid zeros
    for i in range(NC):
        if xI[i] == 0:
            xI[i] = 1e-6
        if xII[i] == 0:
            xII[i] = 1e-6

    # Activity coefficients
    lngamma_I  = lngamma_calc(xI, N_I, T, NC, method, comp, params)
    lngamma_II = lngamma_calc(xII, N_II, T, NC, method, comp, params)
    gammaI = np.maximum(np.exp(lngamma_I), 1e-10)
    gammaII = np.maximum(np.exp(lngamma_II), 1e-10)

    # Objective function
    OF = 0
    for i in range(NC):
        OF += np.sum(N_I)*xI[i]*np.log(xI[i]*gammaI[i]) + np.sum(N_II)*xII[i]*np.log(xII[i]*gammaII[i])

    return OF
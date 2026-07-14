import numpy as np
from scipy.optimize import minimize
import unifac as unifac
import auxfuncs, flash, gibbs_fun, lle_aux
from constants import molecular_weights, densities, pKa_list
from pyswarm import pso
from activity_coeff import calculate_activity_coefficients as lngamma_calc
import sys

def lle_calc_volume(comp, z, T, method, params):
    # Constants
    R = 8.314       # J/(mol*K)
    epsilon = 1e-10 # -

    NC = np.size(comp)

    random_ratios = np.random.rand(len(z))                      # Random values between 0 and 1
    N_I = [ratio * zi for ratio, zi in zip(random_ratios, z)]   # Phase 1
    N_II = [zi - ni for zi, ni in zip(z, N_I)]                  # Phase 2

    # Combine into a single initial guess array
    initial_guess = N_I + N_II
    
    # Gibbs energy minimization
    # Particle Swarm Optimization - PSO
    print('\n Gibbs energy minimization is running, please wait.')
    lb = np.zeros(NC)
    ub = np.concatenate([z])
    bounds = [(lb[i], ub[i]) for i in range(NC)]  # Create a list of tuples
    N, OF = pso(gibbs_fun.gibbs_min, lb, ub, args=(z, T, R, NC, comp, method, params), swarmsize=100, maxiter=200, debug=False)
    N_I = N
    N_II = z - N_I

    # Calculate and print mole fractions
    xI = N_I / np.sum(N_I)
    xII = N_II / np.sum(N_II)

    # Using mole fractions
    z_mole_frac = z / np.sum(z)
    initial_guess = np.concatenate([xI, xII])
    result_flash = flash.flash_algorithm(initial_guess, z_mole_frac, T, R, NC, comp, method, params)
    xI, xII, vv = result_flash

    for i in range(NC):
        N_I[i] = z[i] / (1 + (vv / (1 - vv)) * (xII[i] / xI[i]))
        N_II[i] = z[i] - N_I[i]  # Enforce mass balance

    volume_I = auxfuncs.calculate_phase_volume(N_I, comp, molecular_weights, densities)
    volume_II = auxfuncs.calculate_phase_volume(N_II, comp, molecular_weights, densities)

    return [N_I, N_II, xI, xII, volume_I, volume_II]

def lle_calc(comp, z, T, method, params):
    # Constants
    R = 8.314       # J/(mol*K)
    epsilon = 1e-10 # -

    NC = np.size(comp)

    random_ratios = np.random.rand(len(z))                      # Random values between 0 and 1
    N_I = [ratio * zi for ratio, zi in zip(random_ratios, z)]   # Phase 1
    N_II = [zi - ni for zi, ni in zip(z, N_I)]                  # Phase 2

    # Gibbs energy minimization
    # Particle Swarm Optimization - PSO
    print('\n Checking system\'s stability, please wait.')
    lb = np.zeros(NC)
    ub = np.concatenate([z])
    bounds = [(lb[i], ub[i]) for i in range(NC)]  # Create a list of tuples

    # Combine into a single initial guess array
    x0 = N_I + N_II
    # Checks if feed composition is stable or not
    result_stability = pso(lle_aux.test_stability2, lb, ub, args=(z, T, R, NC, comp, method, params), swarmsize=50, maxiter=100, debug=False)
    OF = result_stability[1]
    print('Done.')
    print(result_stability[0])

    if np.abs(OF)<1e-6:
        OF = 0
    if OF < 0:
        print('System is unstable and will split in 2 phases.')
        flg = 1
    else:
        print('System is stable as it is and won\'t split.')
        flg = 0
    print('OF = ',OF)
    print('________________________________')

    # -----------------------------------------------------------------------------------------------
    # PHASE COMPOSITION CALCULATION
    # -----------------------------------------------------------------------------------------------
    if flg == 0:
        # Calculate the activity coefficients and any other info and end
        print('No need for phase equilibria calculations.')
        xI = z
        xII = z
        return [N_I, N_II, xI, xII, 0]
    else:
        print('Calculating phase composition...')
        # Gibbs energy minimization
        # Particle Swarm Optimization - PSO
        print('\n Gibbs energy minimization is running, please wait.')
        print('\n Calculating phase composition...')
        N, OF = pso(gibbs_fun.gibbs_min, lb, ub, args=(z, T, R, NC, comp, method, params), swarmsize=150, maxiter=200, debug=False)
        print('Done.')
        N_I = N
        N_II = z - N_I

        # Calculate and print mole fractions
        xI = N_I / np.sum(N_I)
        xII = N_II / np.sum(N_II)

        # Using mole fractions
        z_mole_frac = z / np.sum(z)
        initial_guess = np.concatenate([xI, xII])
        print('Running flash algorithm.')
        result_flash = flash.flash_algorithm(initial_guess, z_mole_frac, T, R, NC, comp, method, params)
        print('Done.')
        xI, xII, vv = result_flash

        for i in range(NC):
            N_I[i] = z[i] / (1 + (vv / (1 - vv)) * (xII[i] / xI[i]))
            N_II[i] = z[i] - N_I[i]  # Enforce mass balance

        return [N_I, N_II, xI, xII, vv]

def kow_calc(comp, z, pH, solute, T, method, params, density_method="const"):
    N_I, N_II, xI, xII, vv = lle_calc(comp, z, T, method, params)
    NC = np.size(comp)
    if solute == []:
        print('No solute defined. Please, restart the calculation with a solute defined.')
        sys.exit()
    else:
        pKa = pKa_list[solute]
    
    # TAKE THE LLE RESULT AND CALCULATE Kow
    # First sort the organic and aqueous phases
    water_index = comp.index('WATER')

    # Compare the molar fractions of water in both phases
    if xI[water_index] > xII[water_index]:
        aqueous_phase = 'I'
        xW = xI[:]
        xO = xII[:]
        N_WATER = N_I[water_index]
    else:
        aqueous_phase = 'II'
        xW = xII[:]
        xO = xI[:]
        N_WATER = N_II[water_index]

    # Molar Concentration in the organic phase
    CO = auxfuncs.calculate_molar_concentration(xO, densities, molecular_weights, comp, T_K=T, density_method=density_method)
    # Molar Concentration in the aqueous phase    
    CW = auxfuncs.calculate_molar_concentration(xW, densities, molecular_weights, comp, T_K=T, density_method=density_method)

    # Get the index of the solute
    i_solute = comp.index(solute)
    # Calculate Kow
    Kow = CO * xO[i_solute] / (CW * xW[i_solute])

    # pH correction
    alpha = 1/(10**(-(pH - pKa)) + 1)
    D = (1 - alpha) * Kow

    return [N_I, N_II, xI, xII, vv, Kow, D, CO, CW]

def kowIDAC(solute, T, method, params, density_method="const"):
    comp = ['1-OCTANOL', 'WATER', solute]
    NC = np.size(comp)
    if solute == []:
        print('No solute defined. Please, restart the calculation with a solute defined.')
    else:
        pKa = pKa_list[solute]
    
    # 1-Octanol Molar Concentration in the octanol-rich phase
    CO = 8.37 # mol/L
    # Water Molar Concentration in the aqueous phase    
    CW = 55.5 # mol/L

    # Molar Concentration in the organic phase
    # COteste = auxfuncs.calculate_molar_concentration([0.725, 0.275, 1e-6], densities, molecular_weights, comp, T_K=T)
    COteste = auxfuncs.calculate_molar_concentration([0.725, 0.275, 1e-6], densities, molecular_weights, comp, T_K=T, density_method=density_method)

    # Molar Concentration in the aqueous phase    
    # CWteste = auxfuncs.calculate_molar_concentration([1e-6, 0.999, 1e-6], densities, molecular_weights, comp, T_K=T)
    CWteste = auxfuncs.calculate_molar_concentration([1e-6, 0.999, 1e-6], densities, molecular_weights, comp, T_K=T, density_method=density_method)

    # Activity coefficient in octanol-rich phase
    lngammaO = lngamma_calc(x=[0.725, 0.275, 1e-6], N=[], T=T, NC=NC, method=method, params=params, comp=comp)
    gammaO = np.exp(lngammaO)
    # Activity coefficient in water-rich phase
    lngammaW = lngamma_calc(x=[1e-6, 0.999, 1e-6], N=[], T=T, NC=NC, method=method, params=params, comp=comp)
    gammaW = np.exp(lngammaW)
    # Assuming the solute is in the last position in the 'comp' list
    Kow = (CO*gammaW[-1])/(CW*gammaO[-1])
    logKow = np.log10(Kow)

    return logKow
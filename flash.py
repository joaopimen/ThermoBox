import numpy as np
from scipy.optimize import fsolve, minimize, root
import unifac as unifac
from activity_coeff import calculate_activity_coefficients as lngamma_calc

def flash_solve(vv,K,z,T,NC):
    sum = 0
    for i in range(NC):
        sum += (K[i]*z[i])/(1+vv*(K[i]-1))
    return 1-sum

def flash_algorithm(x, z, T, R, NC, comp, method, params): 
    NP  = 2 # Number of phases (only 2 implemented)
    NC = np.size(comp)
    maxiter = 200       # Maximum number of iterations
    tol     = 1e-10
    xI = x[:NC]
    xII = x[NC:]
    for j in range(maxiter):
        if np.any(np.isnan(xI)) or np.any(np.isnan(xII)):
            # input('...')
            raise ValueError(f"NaN encountered in mole fractions: xI={xI}, xII={xII}")

        # Activity coefficients
        lngamma_I  = lngamma_calc(xI, [], T, NC, method, comp, params)
        lngamma_II = lngamma_calc(xII, [], T, NC, method, comp, params)
        gammaI = np.maximum(np.exp(lngamma_I), 1e-10)
        gammaII = np.maximum(np.exp(lngamma_II), 1e-10)

        if any(np.isnan(gammaI)) or any(np.isnan(gammaII)):
            print('gamma NaN')

        # Check for invalid activity coefficients
        if np.any(gammaI <= 0) or np.any(gammaII <= 0):
            raise ValueError(f"Invalid activity coefficients: gammaI={gammaI}, gammaII={gammaII}")

        # Objective function calculation
        K = gammaII / gammaI
        
        # Separated fraction calculation
        v0 = 0.3
        vv = fsolve(flash_solve, v0, args=(K,z,T,NC))[0]
        # print(f"Calculated phase split fraction (vv): {vv}")

        # Check for invalid vv
        flg_vv = 0
        if np.isnan(vv) or vv <= 0 or vv >= 1:
            # raise ValueError(f"Invalid phase fraction (vv) calculated: vv={vv}")
            vv = 0.5
            flg_vv += 1
        if flg_vv == 5:
            raise ValueError(f"Invalid phase fraction (vv) calculated: vv={vv}")
            
        
        # Update mole fractions in each phase
        new_xI = np.zeros(NC)
        new_xII = np.zeros(NC)
        for i in range(NC):
            new_xI[i] = z[i] / ((1 - vv) / K[i] + vv)
            new_xII[i] = (z[i] - vv * new_xI[i]) / (1 - vv)

        # Blend mole fractions
        xI = (xI + new_xI) / 2
        xII = (xII + new_xII) / 2

        # Check convergence
        if np.sum(np.abs(new_xI - xI)) < tol and np.sum(np.abs(new_xII - xII)) < tol:
            print('Converged.')
            break
    else:
        print('Maximum iterations reached without convergence.')

    return xI, xII, vv

## FLASH WITH CONSTRAINTS WORKING! DO NOT CHANGE!
def flash_algorithm_with_constraints(N, z, T, R, NC, comp, method, v0, params): 
    maxiter = 200       # Maximum number of iterations
    tol = 1e-10         # Convergence tolerance

    # Allocate initial guesses to each phase
    N_I = np.array(N[:NC])
    N_II = np.array(N[NC:])
    z_mole_frac = z / np.sum(z)

    def equations(vars, T, NC, method, params):
        N_I = vars[:NC]
        N_II = vars[NC:]
        
        # Calculate mole fractions
        xI = N_I / np.sum(N_I)
        xII = N_II / np.sum(N_II)

        if any(np.isnan(xI)) or any(np.isnan(xII)):
            print('x nan in equations')

        # Activity coefficients
        lngamma_I  = lngamma_calc(xI, N_I, T, NC, method, comp, params)
        lngamma_II = lngamma_calc(xII, N_II, T, NC, method, comp, params)
        gammaI = np.maximum(np.exp(lngamma_I), 1e-10)
        gammaII = np.maximum(np.exp(lngamma_II), 1e-10)
        
        mass_balance = np.zeros(NC)
        isofugacity = np.zeros(NC)
        for i in range(NC):
            mass_balance[i] = np.abs(z[i] - (N_I[i] + N_II[i]))
            isofugacity[i] = np.abs(xI[i]*gammaI[i] - xII[i]*gammaII[i])

        return np.sum(np.concatenate([mass_balance, isofugacity]))

    # Solve the system of equations
    lb = np.zeros(2*NC)
    ub = np.concatenate([z, z])
    bounds = [(lb[i], ub[i]) for i in range(2 * NC)]

    result = minimize(
        fun=equations,  # Objective function
        x0=np.concatenate([N_I, N_II]),          # Initial guess
        args=(T, NC, method, params),  # Additional arguments for the objective function
        method='SLSQP',            # Optimization method (e.g., Sequential Least Squares Programming)
        bounds=bounds,             # Variable bounds
        constraints=[],   # Constraints
        options={'disp': True}     # Display optimization progress
    )

    if not result.success:
        # Calculate mole fractions
        N_I = result.x[:NC]
        N_II = result.x[NC:]
        xI = N_I / np.sum(N_I)
        xII = N_II / np.sum(N_II)

        # Activity coefficients
        lngamma_I  = lngamma_calc(xI, N_I, T, NC, method, comp, params)
        lngamma_II = lngamma_calc(xII, N_II, T, NC, method, comp, params)
        gammaI = np.maximum(np.exp(lngamma_I), 1e-10)
        gammaII = np.maximum(np.exp(lngamma_II), 1e-10)
        mass_balance = np.zeros(NC)
        isofugacity = np.zeros(NC)
        for i in range(NC):
            mass_balance[i] = np.abs(z[i] - (N_I[i] + N_II[i]))
            isofugacity[i] = np.abs(xI[i]*gammaI[i] - xII[i]*gammaII[i])
        if np.sum(mass_balance) < 1e-6 and np.sum(isofugacity) < 1e-6:
            # Extract results
            N_I = result.x[:NC]
            N_II = result.x[NC:]

            # Calculate phase fraction (vv)
            vv = np.sum(N_II) / np.sum(z)

            return N_I, N_II, vv
        else:
            raise ValueError(f"Flash calculation did not converge: {result.message}")

    # Extract results
    N_I = result.x[:NC]
    N_II = result.x[NC:]

    # Calculate phase fraction (vv)
    vv = np.sum(N_II) / np.sum(z)

    return N_I, N_II, vv
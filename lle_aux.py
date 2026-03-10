from activity_coeff import calculate_activity_coefficients as lngamma_calc
import numpy as np

# LLE Stability test
def test_stability(N, z, T, R, NC, comp, method, params):
    # Distribute the initial compositions for both phases
    NC = int(np.size(N)/2)
    NI = N[0:NC]
    NII = N[NC:2*NC]
    # Convert total moles in the feed to mole fractions
    z_mole_frac = z / np.sum(z)
    # Molar fractions
    xI = NI / np.sum(NI)
    xII = NII / np.sum(NII)
    # Avoid zeros
    for i in range(NC):
        if xI[i] == 0:
            xI[i] = 1e-6
        if xII[i] == 0:
            xII[i] = 1e-6
    
    # Change this later. Make a more efficient test.
    for i in range(NC):
        if xI[i]<0 or xII[i]<0 or z[i]<0:
            print(xI,'\n',xII,'\n',z,'\n')
        if xI[i]>1 or xII[i]>1 or z[i]>1:
            print(xI,'\n',xII,'\n',z,'\n')
        if np.isnan(xI[i]) == 1 or np.isnan(xII[i]) == 1 or np.isnan(z[i]) == 1:
            print(xI,'\n',xII,'\n',z,'\n')

    # Calculates the activity coefficient of both phases
    # Activity coefficients
    lngamma_I  = lngamma_calc(xI, [], T, NC, method, comp, params)
    lngamma_II = lngamma_calc(xII, [], T, NC, method, comp, params)
    lngamma_z  = lngamma_calc(z, [], T, NC, method, comp, params)
    gammaI = np.maximum(np.exp(lngamma_I), 1e-10)
    gammaII = np.maximum(np.exp(lngamma_II), 1e-10)
    gammaZ = np.maximum(np.exp(lngamma_z), 1e-10)

    # Compares the Excess Gibbs energy of the feed with possible 2-phase systems
    OF = 0
    for i in range(NC):
        # OF += R*T*(xI[i]*np.log(gammaI[i]) + xII[i]*np.log(gammaII[i]))
        OF += R*T*(np.sum(NI)*xI[i]*np.log(gammaI[i]*xI[i]) + np.sum(NII)*xII[i]*np.log(gammaII[i]*xII[i]))
    for i in range(NC):
        # OF -= R*T*(z[i]*np.log(gammaZ[i]))
        OF -= R*T*(z[i]*np.log(gammaZ[i]*z[i]))
    return OF

# LLE Stability test
def test_stability2(N, z, T, R, NC, comp, method, params):
    # Distribute the initial compositions for both phases
    NI = N
    NII = z-NI
    # Convert total moles in the feed to mole fractions
    z_mole_frac = z / np.sum(z)
    # Molar fractions
    xI = NI / np.clip(np.nansum(NI), 1e-12, None)
    xII = NII / np.clip(np.nansum(NII), 1e-12, None)
    # Avoid zeros
    for i in range(NC):
        if xI[i] == 0:
            xI[i] = 1e-6
        if xII[i] == 0:
            xII[i] = 1e-6
    
    # Change this later. Make a more efficient test.
    # for i in range(NC):
    #     if xI[i]<0 or xII[i]<0 or z[i]<0:
    #         print(xI,'\n',xII,'\n',z,'\n')
    #     if xI[i]>1 or xII[i]>1 or z[i]>1:
    #         print(xI,'\n',xII,'\n',z,'\n')
    #     if np.isnan(xI[i]) == 1 or np.isnan(xII[i]) == 1 or np.isnan(z[i]) == 1:
    #         print(xI,'\n',xII,'\n',z,'\n')

    # Calculates the activity coefficient of both phases
    # Activity coefficients
    lngamma_I  = lngamma_calc(xI, [], T, NC, method, comp, params)
    lngamma_II = lngamma_calc(xII, [], T, NC, method, comp, params)
    lngamma_z  = lngamma_calc(z_mole_frac, [], T, NC, method, comp, params)
    gammaI = np.maximum(np.exp(lngamma_I), 1e-10)
    gammaII = np.maximum(np.exp(lngamma_II), 1e-10)
    gammaZ = np.maximum(np.exp(lngamma_z), 1e-10)

    # Compares the Excess Gibbs energy of the feed with possible 2-phase systems
    OF = 0
    for i in range(NC):
        # OF += R*T*(xI[i]*np.log(gammaI[i]) + xII[i]*np.log(gammaII[i]))
        OF += R*T*(np.sum(NI)*xI[i]*np.log(gammaI[i]*xI[i]) + np.sum(NII)*xII[i]*np.log(gammaII[i]*xII[i]))
    for i in range(NC):
        # OF -= R*T*(z[i]*np.log(gammaZ[i]))
        OF -= R*T*(z[i]*np.log(gammaZ[i]*z_mole_frac[i]))
    return OF
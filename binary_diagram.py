import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from activity_coeff import calculate_activity_coefficients as lngamma_calc

import random
import warnings
warnings.filterwarnings("ignore")

db = pd.read_csv('antoine_coeff.csv')

db.index = db.comp

def binary_T_known(T):

    P_sat = 10**(A - B/(T+C))

    xx = np.arange(0,1.01,0.01)
    x = np.array((xx,1-xx))

    y = np.zeros((NC,len(xx)))
    P = np.zeros(len(xx))

    for i in range(len(xx)):

        lngamma  = lngamma_calc(x[:,i], n, T + 273.15, NC, 'unifac', comp, params)
        gamma = np.maximum(np.exp(lngamma), 1e-10)

        P[i] = sum(x[:,i]*gamma*P_sat)
        y[:,i] = x[:,i]*P_sat*gamma/P[i]

    return x, y, P

def binary_P_known(P):

    def temperature(T,x):

        lngamma  = lngamma_calc(x, n, T + 273.15, NC, 'unifac', comp, params)
        gamma = np.maximum(np.exp(lngamma), 1e-10)

        P_sat = 10**(A - B/(T+C))

        F = sum(x*P_sat*gamma) - P

        return F
   
    xx = np.arange(0,1.01,0.01)
    x = np.array((xx,1-xx))

    y = np.zeros((NC,len(xx)))
    T = np.zeros(len(xx))

    T0 = random.randint(int(max(Tmin)),int(min(Tmax)))

    for i in range(len(xx)):

        T[i] = fsolve(temperature,T0,args=x[:,i])
        P_sat = 10**(A - B/(T[i]+C))

        lngamma  = lngamma_calc(x[:,i], n, T[i] + 273.15, NC, 'unifac', comp, params)
        gamma = np.maximum(np.exp(lngamma), 1e-10)

        y[:,i] = x[:,i]*P_sat*gamma/P

    return x, y, T

var = input('Insira a variável conhecida (P ou T): ')

if var=='P':
    P = float(input('Insira a pressão em bar: '))

elif var=='T':
    T = float(input('Insira a temperatura em °C: '))

else:
    print('Insira um argumento válido.')
    exit()

NC = 2  # binary mixture

n = np.zeros(NC)
A = np.zeros(NC)
B = np.zeros(NC)
C = np.zeros(NC)

Tmin = np.zeros(NC)
Tmax = np.zeros(NC)

comp = ['WATER','ETHANOL']

for i in range(NC):
    # comp.append(input(f'Insira o nome do {i+1}º componente: '))
   
    if comp[i]=='WATER':
        
        print(db.loc[comp[i]].comp[0])

        A[i] = db.loc[comp[i]].A[0]
        B[i] = db.loc[comp[i]].B[0]
        C[i] = db.loc[comp[i]].C[0]

        Tmax[i] = db.loc[comp[i]].TMAX[0]
        Tmin[i] = db.loc[comp[i]].TMIN[0]

    else:

        print(db.loc[comp[i]].comp)

        A[i] = db.loc[comp[i]].A
        B[i] = db.loc[comp[i]].B
        C[i] = db.loc[comp[i]].C

        Tmax[i] = db.loc[comp[i]].TMAX
        Tmin[i] = db.loc[comp[i]].TMIN

from get_unifac_parameters import get_unifac_parameters
params = get_unifac_parameters(comp)

if var=='T':
    x, y, P = binary_T_known(T)

    plt.plot(x[0,:],P)
    plt.plot(y[0,:],P)
    plt.xlabel('Molar fraction [x,y]')
    plt.ylabel('Pressure [mmHg]')
    plt.title('Binary phase diagram')
    plt.show()

else:
    x, y, T = binary_P_known(P*750)

    plt.plot(x[0,:],T)
    plt.plot(y[0,:],T)
    plt.xlabel('Molar fraction [x,y]')
    plt.ylabel('Temperature [°C]')
    plt.title('Binary phase diagram')
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve

# UNIFAC
# Returns the activity coefficients of a mixture of NC components
def unifac_lngamma(x, T, NG, v, Rk, Qk, a):
	NC = len(x)
	v = np.array(v)
	a = np.array(a)
	x = np.maximum(x, 1e-10)
	# X - Molar fraction of group 'm' in the mixture
	X = np.zeros((NG))
	for m in range(NG):
		X[m] = np.sum(np.dot(x,v[:,m]))/np.sum(np.dot(x,v))

	# Xi - molar fraction of group 'm' in component 'i'
	Xi = np.zeros((NC,NG))
	for i in range(NC):
		for m in range(NG):
			Xi[i,m] = v[i,m]*x[i]/np.sum(np.dot(v[i,:],x[i]))

	# THETA - area fraction of group 'm' in the mixture
	THETA = np.zeros((NG))
	for m in range(NG):
		THETA[m] = Qk[m]*X[m]/np.sum(np.dot(Qk,X))

	# THETAi - areac fraction of group 'm' in component 'i'
	THETAi = np.zeros((NC,NG))
	for i in range(NC):
		for m in range(NG):
			THETAi[i,m] = Qk[m]*Xi[i,m]/np.sum(np.dot(Qk,Xi[i,:]))

	# Interaction energy between functional groups (psi)
	psi = np.zeros((NG,NG))
	for m in range(NG):
		for n in range(NG):
			psi[m,n] = np.exp(-a[m,n]/T)

	# Activity Coefficient of group 'k' in the mixture (lnGAMMA)
	lnGAMMA = np.zeros((NG))
	for k in range(NG):
		sum1 = 0
		sum2 = 0
		for m in range(NG):
			sum1 += THETA[m]*psi[m,k]
			sumden = 0
			for n in range(NG):
				sumden += THETA[n]*psi[n,m]
			sum2 += THETA[m]*psi[k,m]/sumden
		# Check for invalid values in `sum1`
		# if sum1 <= 0:
		# 	raise ValueError(f"Invalid input to log in UNIFAC calculation: sum1={sum1}, Qk={Qk[k]}")

		# Safe calculation
		if sum1 <= 0:
			sum1 = 1e-8
		try:
			lnGAMMA[k] = Qk[k] * (1 - np.log(sum1) - sum2)
		except ValueError as e:
			print(f"Error in UNIFAC calculation for component {k}: sum1={sum1}, sum2={sum2}")
			raise
		# lnGAMMA[k] = Qk[k]*(1 - np.log(sum1) - sum2)

	# Activity Coefficient of group 'k' in component 'i' (lnGAMMAi)
	lnGAMMAi = np.zeros((NC,NG))
	for i in range(NC):
		for k in range(NG):
			sum1 = 0
			sum2 = 0
			for m in range(NG):
				sum1 += THETAi[i,m]*psi[m,k]
				sumden = 0
				for n in range(NG):
					sumden += THETAi[i,n]*psi[n,m]
				sum2 += THETAi[i,m]*psi[k,m]/sumden
			lnGAMMAi[i,k] = Qk[k]*(1 - np.log(sum1) - sum2)

	# Component 'i' Area and Volume parameters
	q = np.zeros((NC))
	r = np.zeros((NC))
	for i in range(NC):
		q[i] = np.dot(v[i,:],Qk)
		r[i] = np.dot(v[i,:],Rk)

	# Component 'i' theta
	theta = np.zeros((NC))
	phi = np.zeros((NC))
	l = np.zeros((NC))
	z = 10
	for i in range(NC):
		theta[i] = x[i]*q[i]/np.sum(np.dot(x,q))
		phi[i] = x[i]*r[i]/np.sum(np.dot(x,r))
		l[i] = z/2*(r[i]-q[i])-(r[i]-1)

	# Residual Contribution
	lngammares = np.zeros((NC))
	for i in range(NC):
		sumres = 0
		for k in range(NG):
			sumres += v[i,k]*(lnGAMMA[k] - lnGAMMAi[i,k])
		lngammares[i] = sumres

	# Combinatorial Contribution
	lngammacomb = np.zeros((NC))
	for i in range(NC):
		lngammacomb[i] = np.log(phi[i]/x[i]) + z/2*q[i]*np.log(theta[i]/phi[i]) + l[i] - phi[i]/x[i]*np.sum(np.dot(x,l))

	lngamma = lngammacomb + lngammares

	# if any(np.isnan(np.exp(lngamma))):
	# 	print('gamma unifac nan')

	return lngamma
#! /usr/bin/python3
# -*- conding=UTF-8 -*-
#  .--,       .--,
# ( (  \.---./  ) )
#  '.__/o   o\__.'
#     {=  ^  =}
#      >  -  <
#     /  Zhu  \
#    //  Yong \\
#   //|  Hao  |\\
#   "'\       /'"_.-~^`'-.
#      \  _  /--'         `
#    ___)( )(___
#   (((__) (__))) 

# ref:
# J. Chem. Phys. 93, 1061(1990)
# J. Chem. Theory Comput. 2013, 9, 4595-4972
# https://github.com/amber-jain-group-iitb/Surface_Hopping

import numpy as np
import matplotlib.pyplot as plt

def Potenial_SAC(p):
	
	'''
		Potential and it's derivatives
		Simple avoided crossing:
			V_{11}(x)=A[1-exp(-Bx)]  x>0
			V_{11}(x)=-A[1-exp(-Bx)] x<0
			V_{22}(x)=-V_{11}(x)
			V_{12}(x)=V_{21}(x)=Cexp(-Dx^2)
		p: position
	'''

	A, B, C, D = 0.01, 1.6, 0.005, 1

	V, V_der = np.zeros([2,2]), np.zeros([2,2])

	if p > 0:
		V[0, 0] = A * ( 1 - np.exp(-B*p) )
		V_der[0 ,0] = A * B * np.exp(-B*p)
	elif p == 0:
		V[0, 0] = 0
		V_der[0, 0] = A*B
	else:
		V[0, 0] = -A * ( 1 - np.exp(B*p) )
		V_der[0, 0] = A * B * ( np.exp(B*p) )

	V[1, 1] = -1 * V[0 ,0]
	V[0, 1] = C * np.exp( -D*p**2 )
	V[1, 0] = V[0, 1]

	V_der[1, 1] = -1 * V_der[0, 0]
	V_der[0, 1] = -2 * p * C * D * np.exp( -D*p**2 )
	V_der[1, 0] = V_der[0, 1]

	return V, V_der

def Potenial_DAC(p):
	
	'''
		Potential and it's derivatives
		Dual avoided crossing:
			V_{11}(x)=0
			V_{22}(x)=-Aexp(-Bx^2)+E0
			V_{11}(x)=V_{21}(x)=Cexp(-Dx^2)
		p: position
	'''

	A, B, C, D, E0 = 0.1, 0.28, 0.015, 0.06, 0.05

	V, V_der = np.zeros([2,2]), np.zeros([2,2])

	V[1, 1] = -A * np.exp(-B*p**2) + E0
	V[0, 1] = C * np.exp( -D*p**2 )
	V[1, 0] = V[0, 1]

	V_der[1, 1] = A * B * 2 * p * np.exp(-B*p**2)
	V_der[0, 1] = -2 * p * C * D * np.exp( -D*p**2 )
	V_der[1, 0] = V_der[0, 1]

	return V, V_der	

def Potenial_ECR(p):

	'''
		Potential and it's derivatives
		Extended coupling with reflection
		V_{11}=A; V_{22}=-A
		V_{12}=Bexp(Cx) x<0
		V_{12}=B[2-exp(-Cx)] x>0
	'''

	A, B, C = 6*(10**-4), 0.10, 0.90

	V, V_der = np.zeros([2,2]), np.zeros([2,2])

	V[0, 0] = A; V[1, 1] = -A

	if p < 0:
		V[0, 1] = B * np.exp( C*p )
		V_der[0, 1] = B * C * np.exp( C*p )
	elif p == 0:
		V[0, 1] = B
		V_der[0, 1] = B*C
	else:
		V[0, 1] = B * ( 2 - np.exp( -C*p ) )
		V_der[0, 1] = B * C * np.exp( -C*p )

	V[1, 0] = V[0, 1]

	V_der[1, 0] = V_der[0, 1]

	return V, V_der

def main():

	npoints = 200
	PMin = -10
	PMax = 10

	plist = np.linspace(PMin, PMax, npoints)

	# SAC: simple avoided crossing
	# eig, vector = linalg.eigh()
	#V, V_der = Potenial_SAC(p=1.1)
	#eig_SAC = np.array([ np.linalg.eigh(Potenial_SAC(p)[0])[0] for p in plist ])
	#plt.plot(plist, eig_SAC, linewidth=2, linestyle='-', color='red')
	#plt.show()
	
	# DAC: Dual avoided crossing
	#eig_DAC = np.array([ np.linalg.eigh(Potenial_DAC(p)[0])[0] for p in plist ])
	#plt.plot(plist, eig_DAC, linewidth=2, linestyle='-', color='blue')
	#plt.show()

	# ECR: Extended coupling with reflection
	eig_ECR = np.array([ np.linalg.eigh(Potenial_ECR(p)[0])[0] for p in plist ])
	plt.plot(plist, eig_ECR, linewidth=2, linestyle='-', color='black')
	plt.show()	

if __name__ == "__main__":

	main()

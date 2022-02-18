#! /usr/bin/python3
# -*- conding: utf-8 -*-
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
from scipy.interpolate import splev, splrep, splder
import matplotlib.pyplot as plt
from PotEig import Potenial_SAC, Potenial_DAC, Potenial_ECR

def Calc_dij_force(V, V_der):

	#ref:
	#	github: https://github.com/smparker/mudslide
	#	F^\xi_{ij} = \langle \phi_i | -\nabla_\xi H | \phi_j\rangle
	#	out = np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)
	#   Force: diagonal element
	#   dij  : off-diagonal element
	#   return eig, dij, force

	eig, vector = np.linalg.eigh(V)

	dij = np.zeros([2,2])
	force = np.zeros([2])

	gap = eig[0] - eig[1]

	MatrixDot = np.dot( np.dot( vector, V_der ), vector )

	# dij
	dij[0, 1] =  MatrixDot[0, 1] / gap
	dij[1, 0] = -1.0 * dij[0, 1]

	# force
	force[1] = -1.0 * MatrixDot[1, 1]
	force[0] = -1.0 * MatrixDot[0, 0]

	return eig, dij, force

def Plot(type_, plist):

	factor = 1

	if type_ == "ECR":
		V = np.array([Potenial_ECR(p)[0] for p in plist])
		V_der = np.array([Potenial_ECR(p)[1] for p in plist])
		factor = 1
	if type_ == "SAC":
		V = np.array([Potenial_SAC(p)[0] for p in plist])
		V_der = np.array([Potenial_SAC(p)[1] for p in plist])
		factor = 50
	if type_ == "DAC":
		V = np.array([Potenial_DAC(p)[0] for p in plist])
		V_der = np.array([Potenial_DAC(p)[1] for p in plist])
		factor = 12

	eig = np.array([ Calc_dij_force(V=V[i], V_der=V_der[i])[0] for i in range(len(plist))])
	dij = np.array([ Calc_dij_force(V=V[i], V_der=V_der[i])[1] for i in range(len(plist))])
	force = np.array([ Calc_dij_force(V=V[i], V_der=V_der[i])[2] for i in range(len(plist))])

	# dij
	plt.plot(plist, dij[:, 0, 1]/factor, linewidth=2, linestyle='-', color='black')
	# force
	plt.plot(plist, force[:, 0], linewidth=2, linestyle='-', color='red')
	plt.plot(plist, force[:, 1], linewidth=2, linestyle=':', color='red')
	# eig
	plt.plot(plist, eig[:, 0], linewidth=2, linestyle='-', color='blue')
	plt.plot(plist, eig[:, 1], linewidth=2, linestyle=':', color='blue')
	
	plt.legend(["dij/%d" % factor, 'force1', 'force2', 'eig1', 'eig2'], loc=1, fontsize='large')

	plt.show()	

def main():

	npoints = 200
	PMin    = -10
	PMax    = 10
	plist = np.linspace(PMin, PMax, npoints)

	type_ = "SAC" # ECR, SAC, DAC

	# plot	
	# dij of factor = 50 for SAC, 12 for DAC, 1 for ECR 
	Plot(type_=type_, plist=plist)


if __name__ == "__main__":
	main()
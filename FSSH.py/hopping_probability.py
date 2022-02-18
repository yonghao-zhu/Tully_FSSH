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
import random
from PotEig import Potenial_SAC, Potenial_DAC, Potenial_ECR
from NACsForce import Calc_dij_force

def HoppingProbability(R_dot, nstate, cstate, mass, Eig, dij, t_step):

	'''
		ref:
		J. Chem. Theory Comput. 2013, 9, 4595-4972
	'''

	#reshape cstate, form (2,) to (1, 2)
	cstate_ = cstate.reshape(1, 2)

	# a_{k j}=c_{k} c_{j}^{*}
	a = np.dot( cstate_.T, cstate_.conjugate() )

	# b_{k l}=2 \hbar^{-1} \operatorname{Im}\left(a_{k l}^{*} V_{k l}\right)-
	#         2 \operatorname{Re}\left(a_{k l}^{*} \mathbf{R} \cdot \mathbf{d}_{k l}\right)
	# Eig.shape = (2,2)
	Eig_ = np.array([[ Eig[0], 0 ], [ 0, Eig[1] ]])

	#b.shape = (2,2)
	b = 2 * ( ( np.multiply( a.conjugate(), Eig_ ).imag ) -\
	          ( R_dot * np.multiply( a.conjugate(), dij ).real ) )

	# produce a random number
	rnd = random.random()
	hop = 0
	gap = Eig[1] - Eig[0]

	# 1/2*M*R_dot^2 > gag
	# g_{kj=}\frac{ \Delta t b_{jk} }{ a_{kk} }
	if ( nstate == 1 ) & ( (0.5 * mass * R_dot**2) > gap ):
		prob_hop = ( t_step * b[1, 0] ) / (a[0, 0].real)
		if prob_hop > rnd: # hop: state1 (nstate) --> state2 (newstate)
			newstate = 2
			hop = 1
	elif nstate == 2:
		prob_hop = ( t_step * b[0, 1] ) / (a[1, 1].real)
		if prob_hop > rnd: # hop: state2 (nstate) --> state1 (newstate)
			newstate = 1
			hop = 1

	# energy conservation
	# \frac{1}{2} M \dot{R}_{2}^{2} + \varepsilon_{fin}=\varepsilon_{ini} + \frac{1}{2} M \dot{R}_{1}^{2}
	if hop == 1:
		R_dot = np.sign(R_dot) * np.sqrt( 2.0 / mass * ( Eig[nstate-1] - Eig[newstate-1] + \
												                 0.5 * mass * R_dot**2 ) )
		nstate = newstate

	return nstate, R_dot

def VelocityVerlet(x, R_dot, mass, force, t_step, nstate):

	'''
		ref:
		J. Chem. Theory Comput. 2013, 9, 4595-4972
	'''
	
	# accelerated speed: acc_t1 = F/M
	acc_t1 = force[nstate-1] / mass

	# Newtonian mechanics: x = x_0 + V*t + 1/2*a*t^2
	x = x + R_dot * t_step + 0.5 * acc_t1 * t_step**2

	V, V_der = Potenial_SAC(p=x)
	Eig, dij, force_t2 = Calc_dij_force(V=V, V_der=V_der)

	# velocity verley algorithm: 
	# \mathbf{v}_{i}(t+\Delta t) = \mathbf{v}_{i}(t) + 
	#                              \frac{1}{2} \Delta t\left[\mathbf{a}_{i}(t) +
	#                              \mathbf{a}_{i}(t+\Delta t)\right]
	acc_t2 = force_t2[nstate-1] / mass
	R_dot = R_dot + 0.5 * t_step * ( acc_t1 + acc_t2 )

	return x, R_dot, V, force_t2, Eig, dij

def main():

	momentum = 1
	mass = 2.0e3
	R_dot = momentum/mass

	cstate = np.array([complex(1,0),complex(0,0)])

	p = -9.9999

	V, V_der = Potenial_ECR(p)

	Eig, dij, force = Calc_dij_force(V=V, V_der=V_der)

	t_step = 1

	print(R_dot)

	nstate, R_dot = HoppingProbability(R_dot=R_dot, cstate=cstate, mass=mass, 
		                               nstate=1, Eig=Eig, t_step=t_step, x=p, dij=dij)
	print(nstate)
	print(R_dot)

if __name__ == "__main__":
	main()
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
from PotEig import Potenial_SAC
from NACsForce import Calc_dij_force

def Coefficient_DiffEqu(R_dot, cstate, Eig, dij):
	'''
	ref: Differential Equation
		i \hbar \dot{c}_{k}=\sum_{j} c_{j}\left(V_{k j}-i \hbar \dot{\mathbf{R}} \cdot \mathbf{d}_{k j}\right)
	for two states (\hbar = 1):
		\dot{c}_{1}= -i \times c_{1} \times \varepsilon _{1} - c_{2} \times \dot{\mathbf{R}} \cdot \mathbf{d}_{12}
		\dot{c}_{2}= -i \times c_{2} \times \varepsilon _{2} - c_{1} \times \dot{\mathbf{R}} \cdot \mathbf{d}_{21}
	
	Eig.shape = (2,)
	dij.shape = (2, 2)
	cstate.shape = (1, 2) dtype=complex

	R_dot = momentum/mass, real
	
	combine = [[E1          -i*R_dot*d21]
			   [-i*R_dot*d12 E2         ]]
	c_dot = -1j * np.dot(cstate, combine)

	return c_dot
	'''
	combine = np.zeros([2, 2], dtype=complex)

	combine[0, 0] = Eig[0]; combine[1, 1] = Eig[1]

	combine[0, 1] = -1j*R_dot*dij[1, 0]

	combine[1, 0] = -1j*R_dot*dij[0, 1]

	c_dot = -1j * np.dot(cstate, combine) 

	return c_dot


def Fourth_RungeKutta(R_dot, cstate, Eig, dij, t_step):
	'''
		propagate electronic state forward
		ref: 4th Runge-Kutta method (google or baidu)
		y_{n+1} = y_n + \frac{h}{6}(K_1 + 2K_2 + 2K_3 + K_4)
		K_1 = f(x_n, y_n)
		K_2 = f(x_n + \frac{h}{2}, y_n + \frac{h}{2}K_1)
		K_3 = f(x_n + \frac{h}{2}, y_n + \frac{h}{2}K_2)
		K_4 = f(x_n + h, y_n + hK_3)
	'''

	# y_n
	cstate_1 = cstate[:]
	c_dot_1 = Coefficient_DiffEqu(R_dot=R_dot, cstate=cstate_1, 
                                  Eig=Eig, dij=dij)

	# h * K_1
	K_1 = t_step * c_dot_1

	# h * K_2, x = x_n + h/2
	cstate_2 = cstate_1 + K_1/2.0
	c_dot_2 = Coefficient_DiffEqu(R_dot=R_dot, cstate=cstate_2, 
								  Eig=Eig, dij=dij)
	K_2 = t_step * c_dot_2

	# h * K_3, x = x_n + h/2
	cstate_3 = cstate_1 + K_2/2.0
	c_dot_3 = Coefficient_DiffEqu(R_dot=R_dot, cstate=cstate_3, 
								  Eig=Eig, dij=dij)
	K_3 = t_step * c_dot_3

	# h * K_4, x = x_n + h
	cstate_4 = cstate_1 + K_3
	c_dot_4 = Coefficient_DiffEqu(R_dot=R_dot, cstate=cstate_4, 
							      Eig=Eig, dij=dij)
	K_4 = t_step * c_dot_4

	# y_{n+1}, h is in K
	cstate_f = cstate_1 + ( K_1 + 2*K_2 + 2*K_3 + K_4 ) / 6

	return cstate_f


def main():

	momentum = 1
	mass = 2.0e3
	R_dot = momentum/mass
	npoints = 200
	PMin    = -10
	PMax    = 10
	plist = np.linspace(PMin, PMax, npoints)
	cstate = np.array([complex(1,0),complex(0,0)])

	V,V_der = Potenial_SAC(-9.9999)

	Eig, dij, force = Calc_dij_force(V=V, V_der=V_der)

	t_step = 1

	cstate_f = Fourth_RungeKutta(R_dot=R_dot, cstate=cstate, 
					  Eig=Eig, dij=dij, t_step=t_step)

	print(cstate_f)

if __name__ == "__main__":
	main()






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

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from PotEig import Potenial_SAC, Potenial_DAC, Potenial_ECR
from NACsForce import Calc_dij_force
from Fourth_RungeKutta import Fourth_RungeKutta
from hopping_probability import VelocityVerlet, HoppingProbability

def main():

	ntraj  = 1
	t_step = 1
	mass   = 2.0e3
	PMin   = -10
	PMax   = 10
	npoints= 200

	momentums, prob11, prob12, prob22 = [], [], [], []

	for momentum in range(20, 21):

		j, k = [0, 0], [0, 0]

		#------------------------------------
		for nt in range(0, ntraj):

			p      = -9.9999
			R_dot  = momentum / mass
			nstate = 1
			cstate = np.array([complex(1,0), complex(0,0)])

			V, V_der = Potenial_SAC(p=p)
			Eig, dij, force    = Calc_dij_force(V=V, V_der=V_der)

			#-------------------------------------------
			# propagate
			n = 0
			while ( p > PMin and p < PMax ):

				# Eletronic state: Coefficient Differential Equation
				cstate = Fourth_RungeKutta(R_dot=R_dot, cstate=cstate, Eig=Eig, dij=dij, t_step=t_step)

				# Hopping Probability
				nstate, R_dot = HoppingProbability(R_dot=R_dot, cstate=cstate, mass=mass,
												   t_step=t_step, Eig=Eig, dij=dij, nstate=nstate)

				# atomic motion (MD): velocity verley algorithm
				p, R_dot, V, force, Eig, dij = VelocityVerlet(x=p, R_dot=R_dot, mass=mass, force=force, 
					                                          t_step=t_step, nstate=nstate)
				
				print('-n-->', n)
				print('-nstate-->', nstate)
				print('-R_dot-->', R_dot)
				print('-p-->', p)
				print('-cstate-->', cstate, "\n")
				n += 1
				if n == 1038:
					break
				
			print('finally n-->', n)
			print('nstate-->', nstate)
			#-------------------------------------------

			# Statistical trajectories
			if nstate == 1:
				if p < 0:
					j[0] = j[0] + 1
				if p > 0:
					j[1] = j[1] + 1
			else:
				if p < 0:
					k[0] = k[0] + 1
				if p > 0:
					k[1] = k[1] + 1
		#----------------------------------------

		prob_trans_lower =float(j[1]/ntraj)
		prob_refl_lower  =float(j[0]/ntraj)
		prob_trans_upper =float(k[1]/ntraj)

		momentums.append(momentum)
		prob12.append(prob_trans_lower)
		prob11.append(prob_refl_lower)
		prob22.append(prob_trans_upper)
		print("momentum-->",momentum,"**",j,k, "--- %s seconds ---" % ( time.time()-start_time) )

	#Plotting Probabilities
	# with open('momentum-prob.txt', 'w+') as f:
	# 	for i in range(len(momentums)):
	# 		f.writelines(str(momentums[i]) + \
	# 			'    '+ str(prob11[i]) + \
	# 			'    '+ str(prob12[i]) + \
	# 			'    '+ str(prob22[i]) + \
	# 			'\n')

	fig, ax = plt.subplots(3, sharex=True, gridspec_kw={'hspace': 0.1})

	ax[2].set_ylim(None,1)
	ax[0].plot(momentums, prob12, label = "prob_trans_lower", linestyle = "-", marker = "o")
	ax[1].plot(momentums, prob11, label = "prob_refl_lower",  linestyle = "-", marker = "v")
	ax[2].plot(momentums, prob22, label = "prob_trans_upper", linestyle = "-", marker = "s")

	ax[2].set(xlabel='momentum')	
	ax[0].legend(); ax[1].legend(); ax[2].legend();
	# plt.savefig('probability.png', dpi=600, format='png')
	#plt.show()

if __name__ == "__main__":
	start_time = time.time()
	main()

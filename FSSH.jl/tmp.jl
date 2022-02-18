# Julia1.6.5
push!(LOAD_PATH, ".")

using PotEig
using Fourth_RungeKutta
include("NACsForce.jl")
include("HoppingProbability.jl")

# test FourthRK
# V, V_der = Potential_SAC(1.1)
# eig, dij, force = clac_dij_force(V, V_der)
# cstate = [Complex(0.0, 1.0) Complex(0.0, 0.0)]
# R_dot = 1 / 2000
# c = FourthRK(R_dot, cstate, eig, dij, 1)


# debug n = 1036 --> 1037
#t_step = 1
#nstate = 1
#mass = 2000.0
#p = 0.33873462423179684
#R_dot = 0.009803768044093174
#cstate = [-0.46785403-0.44282623im -0.73112232+0.22467241im]
#cstate = [-0.4678540292766279 - 0.44282622833848345im -0.731122321204738 + 0.22467240641495065im]
#V, V_der        = Potential_SAC(p)
#Eig, dij, force = clac_dij_force(V, V_der)
#cstate = FourthRK!(R_dot, cstate, Eig, dij, t_step)
#nstate, R_dot = HoppingProbability!(R_dot, nstate, mass, cstate, Eig, dij, t_step)
#print("-nstate-->", nstate)

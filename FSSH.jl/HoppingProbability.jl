# Julia1.6.5

using PotEig
include("NACsForce.jl")

# ref:
# J. Chem. Phys. 93, 1061(1990)
# J. Chem. Theory Comput. 2013, 9, 4595-4972
# https://github.com/amber-jain-group-iitb/Surface_Hopping

" Hopping Probability "
function HoppingProbability!(R_dot::Float64, nstate::Int64, mass::Float64,
                            cstate::Matrix{Complex{Float64}},
                            Eig::Vector{Float64}, dij::Matrix{Float64},
                            t_step::Int64)

	# ref: J. Chem. Theory Comput. 2013, 9, 4595-4972
	# https://github.com/amber-jain-group-iitb/Surface_Hopping

    # matrix: a_{k j}=c_{k} c_{j}^{*}
	a = transpose(cstate) * conj(cstate)

    # b_{k l}=2 \hbar^{-1} \operatorname{Im}\left(a_{k l}^{*} V_{k l}\right)-
	#         2 \operatorname{Re}\left(a_{k l}^{*} \mathbf{R} \cdot \mathbf{d}_{k l}\right)
    # matrix: b
	Eig_ = [Eig[1] 0.0; 0.0 Eig[2]]
	b = 2.0 .* ( imag(conj(a) .* Eig_) .- R_dot .* real(conj(a) .* dij) )

	# produce a random number
	rnd = rand()
	hop = 0
	gap = Eig[2] - Eig[1]

	# 1/2*M*R_dot^2 > gag
	# g_{kj=}\frac{ \Delta t b_{jk} }{ a_{kk} }
	if ( nstate == 1 ) && ( 0.5 * mass * abs2(R_dot) > gap )
		prob_hop = (t_step * b[2, 1]) / real(a[1, 1])
		if prob_hop > rnd
			newstate = 2
			hop = 1
		end # if
	elseif nstate == 2
		prob_hop = (t_step * b[1, 2]) / real(a[2, 2])
		if prob_hop > rnd
			newstate = 1
			hop = 1
		end # if
	end # if

	# energy conservation
	# \frac{1}{2} M \dot{R}_{2}^{2} + \varepsilon_{fin}=\varepsilon_{ini} + \frac{1}{2} M \dot{R}_{1}^{2}
	if hop == 1
		R_dot = sign(R_dot) * sqrt( 2.0 / mass * ( Eig[nstate]-Eig[newstate] + 0.5 * mass * abs2(R_dot) ) )
		nstate = newstate
	end # if

	return nstate, R_dot

end # function HoppingProbability

" MD: velcity verlet "
function VelocityVerlet!(p::Float64, R_dot::Float64, mass::Float64,
						force::Vector{Float64}, t_step::Int64, nstate::Int64)

	# ref: J. Chem. Theory Comput. 2013, 9, 4595-4972
	# https://github.com/amber-jain-group-iitb/Surface_Hopping

	# accelerated speed: acc_t1 = F/M
	acc_t1 = force[nstate] / mass

	# Newtonian mechanics: x = x_0 + V*t + 1/2*a*t^2
	p_t2 = p + R_dot * t_step + 0.5 * acc_t1 * abs2(t_step)

	V, V_der = Potential_SAC(p_t2)
	Eig, dij, force_t2 = clac_dij_force(V, V_der)

	# velocity verley algorithm:
	# \mathbf{v}_{i}(t+\Delta t) = \mathbf{v}_{i}(t) +
	#                              \frac{1}{2} \Delta t\left[\mathbf{a}_{i}(t) +
	#                              \mathbf{a}_{i}(t+\Delta t)\right]
	acc_t2 = force_t2[nstate] / mass
	R_dot_t2 = R_dot + 0.5 * t_step * ( acc_t1 + acc_t2 )

	return p_t2, R_dot_t2, force_t2, Eig, dij

end # function VelocityVerlet

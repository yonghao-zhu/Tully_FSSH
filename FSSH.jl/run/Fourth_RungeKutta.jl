# Julia1.6.5

module Fourth_RungeKutta

export FourthRK!, CoefficientDiffEqu

" Coefficient differential equation "
function CoefficientDiffEqu(R_dot::Float64,
                            cstate::Matrix{Complex{Float64}},
                            Eig::Vector{Float64},
                            dij::Matrix{Float64})

	# ref: Differential Equation
	# 	i \hbar \dot{c}_{k}=\sum_{j} c_{j}\left(V_{k j}-i \hbar \dot{\mathbf{R}} \cdot \mathbf{d}_{k j}\right)
	# for two states (\hbar = 1):
	# 	\dot{c}_{1}= -i \times c_{1} \times \varepsilon _{1} - c_{2} \times \dot{\mathbf{R}} \cdot \mathbf{d}_{12}
	# 	\dot{c}_{2}= -i \times c_{2} \times \varepsilon _{2} - c_{1} \times \dot{\mathbf{R}} \cdot \mathbf{d}_{21}

	# Eig.shape = (2,)
	# dij.shape = (2, 2)
	# cstate.shape = (1, 2) dtype=complex

	# R_dot = momentum/mass, real

	# combine = [[E1          -i*R_dot*d21]
	# 		   [-i*R_dot*d12 E2         ]]
	# c_dot = -1j * np.dot(cstate, combine)

	# return c_dot

	combine = zeros(Complex{Float64}, 2, 2)

	combine[1, 1] = Eig[1]; combine[2, 2] = Eig[2]

	combine[1, 2] = -im * R_dot * dij[2, 1]
	combine[2, 1] = -im * R_dot * dij[1, 2]

	c_dot = -im .* (cstate * combine)

	return c_dot

end # function CoefficientDiffEqu

" 4th Runge Kutta for Differential Equation "
function FourthRK!(R_dot::Float64,
	              cstate::Matrix{Complex{Float64}},
				  Eig::Vector{Float64},
				  dij::Matrix{Float64},
				  t_step::Int64)

	# propagate electronic state forward
	# ref: 4th Runge-Kutta method (google or baidu)
	# y_{n+1} = y_n + \frac{h}{6}(K_1 + 2K_2 + 2K_3 + K_4)
	# K_1 = f(x_n, y_n)
	# K_2 = f(x_n + \frac{h}{2}, y_n + \frac{h}{2}K_1)
	# K_3 = f(x_n + \frac{h}{2}, y_n + \frac{h}{2}K_2)
	# K_4 = f(x_n + h, y_n + hK_3)

	# y_n
	cstate_1 = cstate
	c_dot_1 = CoefficientDiffEqu(R_dot, cstate_1, Eig, dij)

	# h*K_1
	K_1 = t_step * c_dot_1

	# h*K_2, x = x_n + h/2
	cstate_2 = cstate_1 + K_1 / 2.0
	c_dot_2 = CoefficientDiffEqu(R_dot, cstate_2, Eig, dij)
	K_2 = t_step * c_dot_2

	# h*K_3, x = x_n + h/2
	cstate_3 = cstate_1 + K_2 / 2.0
	c_dot_3 = CoefficientDiffEqu(R_dot, cstate_3, Eig, dij)
	K_3 = t_step * c_dot_3

	# h*K_4, x = x_n + h
	cstate_4 = cstate_1 + K_3
	c_dot_4 = CoefficientDiffEqu(R_dot, cstate_4, Eig, dij)
	K_4 = t_step * c_dot_4

	# y_{n+1}, h is in K
	cstate_f = cstate_1 + ( K_1 + 2.0*K_2 + 2.0*K_3 + K_4 ) / 6.0

	return cstate_f

end # fucntion FourthRK

end  # module Fourth_RungeKutta

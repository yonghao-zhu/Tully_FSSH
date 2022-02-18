# Julia1.6.5

# ref:
# J. Chem. Phys. 93, 1061(1990)
# J. Chem. Theory Comput. 2013, 9, 4595-4972
# github:
# https://github.com/amber-jain-group-iitb/Surface_Hopping
# https://github.com/smparker/mudslide

# matrix operator
using LinearAlgebra

function clac_dij_force(V::Matrix{Float64}, V_der::Matrix{Float64})

    # F^\xi_{ij} = \langle \phi_i | -\nabla_\xi H | \phi_j\rangle
    # out = np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)
    # Force: diagonal element
    # dij  : off-diagonal element
    # return eig, dij, force

    eig = convert( Vector{Float64}, eigvals(V) )
    vector = convert( Matrix{Float64}, eigvecs(V) )

    dij, force = zeros(Float64, 2, 2), zeros(Float64, 2)

    MatrixDot = ( vector * V_der ) * vector

    # dij matrix
    dij[1, 2] = MatrixDot[1, 2] / ( eig[1] - eig[2] )
    dij[2, 1] = -1.0 * dij[1, 2]

    # force matrix
    force[2] = -1.0 * MatrixDot[2, 2]
    force[1] = -1.0 * MatrixDot[1, 1]

    return eig, dij, force

end # calc-dij_force

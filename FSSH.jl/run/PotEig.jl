# Julia1.6.5
# ref: J. Chem. Phys. 93, 1061(1990)

module PotEig

export Potential_SAC, Potential_DAC, Potential_ECR

# potential SAC
function Potential_SAC(p::Float64)

    # Potential and it's derivatives
    # Simple avoided crossing:
    #     V_{11}(x)=A[1-exp(-Bx)]  x>0
    #     V_{11}(x)=-A[1-exp(-Bx)] x<0
    #     V_{22}(x)=-V_{11}(x)
    #    V_{12}(x)=V_{21}(x)=Cexp(-Dx^2)
    # p: position --> x

    A, B, C, D = 0.01, 1.6, 0.005, 1.0 # Float64
    V, V_der = zeros(Float64, 2, 2), zeros(Float64, 2, 2)

    if p > 0.0
        V[1, 1] = A * ( 1.0 - exp( -B*p ) )
        V_der[1, 1] = A * B * exp( -B*p )
    elseif p == 0.0
        V[1, 1] = 0.0
        V_der[1, 1] = A * B
    else # p < 0
        V[1, 1] = -A * ( 1.0 - exp( B*p ) )
        V_der[1, 1] = A * B * exp( B*p )
    end # if

    # potential matrix
    V[2, 2] = -1.0 * V[1, 1]
    V[1, 2] = C * exp( -D*abs2(p) )
    V[2, 1] = V[1, 2]

    # potential differential
    V_der[2 ,2] = -1.0 * V_der[1, 1]
    V_der[1, 2] = -2.0 * p * C * D * exp( -D*abs2(p) )
    V_der[2, 1] = V_der[1, 2]

    return V, V_der
end # Potential_SAC

# potential DAC
function Potential_DAC(p::Float64)

    # Potential and it's derivatives
    # Dual avoided crossing:
    #     V_{11}(x)=0
    #     V_{22}(x)=-Aexp(-Bx^2)+E0
    #     V_{11}(x)=V_{21}(x)=Cexp(-Dx^2)
    # p: position --> x

    A, B, C, D, E0 = 0.1, 0.28, 0.015, 0.06, 0.05
    V, V_der = zeros(Float64, 2, 2), zeros(Float64, 2, 2)

    # potential matrix
    V[2, 2] = -A * exp( -B*abs2(p) ) + E0
    V[1, 2] = C * exp( -D*abs2(p) )
    V[2, 1] = V[1, 2]

    # potential differential
    V_der[2, 2] = A * B * 2.0 * p * exp( -B*abs2(p) )
    V_der[1, 2] = -2.0 * p * C * D * exp( -D*abs2(p) )
    V_der[2, 1] = V_der[1, 2]

    return V, V_der
end # Potential_DAC

# potential ECR
function Potential_ECR(p::Float64)

    # Potential and it's derivatives
    # Extended coupling with reflection
    # V_{11}=A; V_{22}=-A
    # V_{12}=Bexp(Cx) x<0
    # V_{12}=B[2-exp(-Cx)] x>0

    A, B, C = 0.0006, 0.10, 0.90
    V, V_der = zeros(Float64, 2, 2), zeros(Float64, 2, 2)

    V[1, 1] = A; V[2, 2] = -A
    if p < 0
        V[1, 2] = B * exp( C*p )
        V_der[1, 2] = B * C * exp( C*p )
    elseif p == 0
        V[1, 2] = B
        V_der[1, 2] = B * C
    else
        V[1 ,2] = B * ( 2.0 - exp( -C*p ) )
        V_der[1, 2] = B * C * exp( -C*p )
    end # if

    V[2, 1] = V[1, 2]

    V_der[2, 1] = V_der[1, 2]

    return V, V_der

end # Potential_ECR

end # PotEig

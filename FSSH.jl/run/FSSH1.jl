# Julia1.6.5
push!(LOAD_PATH, ".")

# ref:
# J. Chem. Phys. 93, 1061(1990)
# J. Chem. Theory Comput. 2013, 9, 4595-4972
# github:
# https://github.com/amber-jain-group-iitb/Surface_Hopping
# https://github.com/smparker/mudslide

using PotEig
using Fourth_RungeKutta
include("NACsForce.jl")
include("HoppingProbability.jl")

function FSSH(a::Int64)

    # parameters
    ntraj   = 2000
    ms      = 30
    t_step  = 1
    mass    = 2000.0
    PMin    = -10.0
    PMax    = 10.0

    # creat data collector
    momentums, prob11, prob12, prob22 = [], [], [], []

    # main loop
    for momentum in 1:ms

        j, k =  [0 0],[0 0]

        #-----------------------------------------------------------
        for nt in 1:ntraj

            # initial parameters
            p      = -9.9999
            R_dot  = convert(Float64, momentum) / mass
            nstate = 1
            cstate = [ 1.0+0.0im 0.0+0.0im ]

            V, V_der        = Potential_SAC(p)
            Eig, dij, force = clac_dij_force(V, V_der)

            #*****************************************************
            # propagate
            n = 0
            while p > PMin && p < PMax

                # Eletronic state: Coefficient Differential Equation
                cstate = FourthRK!(R_dot, cstate, Eig, dij, t_step)

                # Hopping Probability
                nstate, R_dot = HoppingProbability!(R_dot, nstate, mass, cstate, Eig, dij, t_step)

                # atomic motion (MD): velocity verley algorithm
                p, R_dot, force, Eig, dij = VelocityVerlet!(p, R_dot, mass, force, t_step, nstate)

            end # while
            #*****************************************************

            # Statistical trajectories
            if nstate == 1
                if p < 0.0
                    j[1] += 1
                else # p > 0.0
                    j[2] += 1
                end # if
            else
                if p < 0.0
                    k[1] += 1
                else # p > 0.0
                    k[2] += 1
                end # if
            end # if

        end # for
        #-----------------------------------------------------------

    append!(momentums, momentum)
    append!(prob11, j[1]/ntraj)
    append!(prob12, j[2]/ntraj)
    append!(prob22, k[2]/ntraj)

    println("momentum--> ", momentum, " ** ", j, k, " ** ", "\n")
    end # for

    # save file
    open("momentums.txt", "w+") do f
        write(f, "momentum", "  prob11", "  prob12", "  prob22\n")
        for i in 1: ms
            write(f, string(i), "  ", string(prob11[i]), "  ")
            write(f, string(prob12[i]), "  ", string(prob22[i]))
            write(f, "\n")
        end
    end

end # function

#using BenchmarkTools

FSSH(1)

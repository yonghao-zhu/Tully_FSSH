# Julia1.6.5
push!(LOAD_PATH, ".")
# matrix operator
using LinearAlgebra
# plot
using Plots
# potential and eigenvalue
using PotEig: Potential_SAC, Potential_DAC, Potential_ECR
include("NACsForce.jl")
#---------------------------------------------------------
npoints = 200
PMin = -10.0
PMax = 10.0
plist = range(PMin, stop=PMax, length=npoints)

# eigenvalue --> SAC, DAC, and ECR
eig = zeros(Float64, size(plist)[1], 2)
for i in 1:size(plist)[1]
    eig[i, :] = eigvals(Potential_ECR(plist[i])[1])
end

# dij and force
dij = zeros(Float64, size(plist)[1], 2, 2)
force = zeros(Float64, size(plist)[1], 2)
for i in 1: size(plist)[1]
    V_, V_der_ = Potential_DAC(plist[i])
    eig_, dij_, force_ = clac_dij_force(V_, V_der_)
    dij[i, :, :] = dij_
    force[i, :] = force_
end

eig = dij[:, 1, 2]

# plot
eig_min = round(findmin(eig)[1], digits=3)
eig_max = round(findmax(eig)[1], digits=3)

plot(plist, eig, linewidth=4,
    title="dij", xticks=range(PMin, stop=PMax, length=5),
    yticks=range(eig_min, stop=eig_max, length=5),
    xlabel="position", ylabel="Potential",
    label=["eig1" "eig2"])

savefig("xx.png")

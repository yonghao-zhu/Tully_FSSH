#! /bin/bash
#PBS -l nodes=node8:ppn=1
#PBS -N zyh_FSSH.jl
#PBS -o stand.log
#PBS -e stand.err
#PBS -q long

cd $PBS_O_WORKDIR

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export PATH=$PATH:/data/long08/soft/julia.1.6.5/julia-1.6.5/bin

julia FSSH1.jl > log


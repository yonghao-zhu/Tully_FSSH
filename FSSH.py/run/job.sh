#! /bin/bash
#PBS -l nodes=node8:ppn=1
#PBS -N zyh_FSSH4
#PBS -o stand.log
#PBS -e stand.err
#PBS -q long

cd $PBS_O_WORKDIR

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export PATH=/opt/longrun/anaconda3/bin:$PATH

python FSSH.py > log.dat


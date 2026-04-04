#!/bin/bash
#SBATCH --job-name=poiss2d_job       
#SBATCH --output=poiss2d_%j.out      
#SBATCH --error=poiss2d_%j.err       
#SBATCH --nodes=1                    
#SBATCH --ntasks=4                   
#SBATCH --time=00:05:00              
#SBATCH --partition=compute          

#Load modules
module purge
module load mpi/latest
module load tbb/latest
module load compiler-rt/latest
module load oclfpga/latest
module load compiler/latest

#Call make
make clean
make poiss2d

#RUn job
mpirun -np 4 ./poiss2d 31
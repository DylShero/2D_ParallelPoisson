#!/bin/bash
#SBATCH --job-name=mpi_io_job        
#SBATCH --output=mpi_io_%j.out       
#SBATCH --error=mpi_io_%j.err        
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


make clean
make

#Run
mpirun -np 4 ./mpi_io_test
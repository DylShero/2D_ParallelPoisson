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

echo "TEST 1: Grid Size 15x15 on 4 Processors"
mpirun -np 4 ./poiss2d 15 1
#Move output to new file
mv global_solution.txt global_solution_15.txt 

echo "TEST 2: Grid Size 31x31 on 4 Processors"
mpirun -np 4 ./poiss2d 31 1
mv global_solution.txt global_solution_31.txt 

echo "PERFORMANCE COMPARISON: 16 Processors (Grid 31x31)"
echo ">>> Testing Method 1: SendRecv"
mpirun -np 16 ./poiss2d 31 1

echo -e "\n>>> Testing Method 2: Non-Blocking"
mpirun -np 16 ./poiss2d 31 2
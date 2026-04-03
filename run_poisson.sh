#!/bin/bash
#SBATCH --job-name=poiss1d_job       # Name of the job
#SBATCH --output=poiss1d_%j.out      # Output file name (%j gets replaced with the Job ID)
#SBATCH --error=poiss1d_%j.err       # Error log file name
#SBATCH --nodes=1                    # Number of nodes to request
#SBATCH --ntasks=4                   # Number of MPI processes (ranks) to run
#SBATCH --cpus-per-task=1            # Number of CPU cores per MPI process
#SBATCH --time=00:05:00              # Maximum execution time 
#SBATCH --partition=compute          # Partition/queue to submit to

#Load modules
module load tbb/2021.12
module load compiler-rt/2024.1.0
module load gcc/13.2.0-gcc-8.5.0-sfeapnb
module load mkl/2024.1 

echo "Compiling poiss1d..."
make poiss1d

#passing '31' as the grid size
echo "Running poiss1d on 31x31 grid..."
mpirun -np $SLURM_NTASKS ./poiss1d 31
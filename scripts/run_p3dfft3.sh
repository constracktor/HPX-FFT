#!/usr/bin/bash
#SBATCH --output=output/output_p3dfft_%j.log
#SBATCH --error=output/error/error_p3dfft_%j.log
#SBATCH --ntasks=4
#SBATCH --partition=buran
#SBATCH --nodes=2
#SBATCH --cpus-per-task=24

module load openmpi
#                                                  threads N_X N_Y N_Z ndim header
srun --mpi=pmix ./examples/fftw/build/p3dfft_mpi_omp 4      512 512 512 2    1
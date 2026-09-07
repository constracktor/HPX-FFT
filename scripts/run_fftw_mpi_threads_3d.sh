#!/usr/bin/bash
#SBATCH --output=output/output_fftw_mpi_threads_3d_%j.log      # Standard output log
#SBATCH --error=output/error_fftw_mpi_threads_3d_%j.log        # Error log file
#SBATCH --ntasks=4
#SBATCH --partition=buran
#SBATCH --nodes=2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4

module load openmpi
srun --mpi=pmix ./examples/fftw/build/fftw_mpi_threads_3d 4 256 256 256 estimate 1

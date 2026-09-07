#!/usr/bin/bash
#SBATCH --output=output/output_fftw_threads_3d_%j.log      # Standard output log
#SBATCH --error=output/error_fftw_threads_3d_%j.log        # Error log file
#SBATCH --ntasks=1
#SBATCH --partition=buran
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

srun ./examples/fftw/build/fftw_threads_3d 4 256 256 256 estimate 1

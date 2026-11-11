#!/usr/bin/bash
#SBATCH --output=output/output_hpxfft_distributed_loop_3d_%j.log        # Standard output log
#SBATCH --error=output/error/error_hpxfft_distributed_loop_3d_%j.log      # Error log file
#SBATCH --ntasks=4
#SBATCH --partition=buran
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=48
export PMIX_MCA_psec=native
srun --mpi=pmix ./examples/hpxfft/build/hpxfft_distributed_loop_3d_pencil --nx=512 --ny=512 --nz=510 --plan=measure --header=true --run=scatter --hpx:ini=hpx.parcel.mpi.enable=1 --hpx:ini=hpx.parcel.tcp.enable=0 --hpx:ini=hpx.parcel.lci.enable=0 --result=false

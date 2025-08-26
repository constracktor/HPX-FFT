#!/bin/bash
#SBATCH --job-name=hpx_fft_job          # Job name
#SBATCH --output=hpx_fft_job.log        # Standard output and error log
#SBATCH --mail-type=NONE                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alexander.strack@ipvs.uni-stuttgart.de       # Where to send mail	
#SBATCH --time=10:00:00                 # Time limit hrs:min:sec
#SBATCH --exclusive                     # Exclusive ressource access
# optional
#SBATCH --partition=buran               # Name of partition
#SBATCH --nodes=16                      # Number of nodes
#SBATCH --ntasks=16                     # Number of MPI ranks
#SBATCH --cpus-per-task=48              # Number of cores per MPI rank 

# Benchmark script for distributed memory strong scaling
# $1: Executable name
# $2: FFTW planning flag (estimate/measure)
# $3: Number of nodes
# $4: HPX collective (scatter/all_to_all)
# $5: Number of runs
# Parameters
LOOP=$5
POW_START=1
POW_STOP=$3
BASE_SIZE=16384
# Get run command
COMMAND="srun -N 1 -n 1"
EXECUTABLE="../$1"
ARGUMENTS="--nx=$BASE_SIZE --ny=$BASE_SIZE --plan=$2 --run=$4"
# Log Info
pwd; hostname; date
# Create directories to store data
mkdir result
mkdir plans
# Strong scaling loop from 2^pow_start to 2^pow_stop nodes
$COMMAND $EXECUTABLE $ARGUMENTS --header=true 
for (( j=1; j<$LOOP; j=j+1 ))
do
    $COMMAND $EXECUTABLE $ARGUMENTS
done
for (( i=2**$POW_START; i<=2**$POW_STOP; i=i*2 ))
do
    COMMAND="srun -N $i -n $i"
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        $COMMAND $EXECUTABLE $ARGUMENTS
    done
done
# Log Info
date
#!/bin/bash
#SBATCH --job-name=fftw_job          # Job name
#SBATCH --output=fftw_job.log        # Standard output
#SBATCH --error=fftw_error_%A.log    # Error Log
#SBATCH --mail-type=NONE                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --time=2:00:00                 # Time limit hrs:min:sec
#SBATCH --exclusive                     # Exclusive ressource access

# Benchmark script for shared memory strong scaling
# $1: Executable name
# $2: FFTW planning flag (estimate/measure)
# $3: Starting number of cores
# $4: Stopping number of cores
# $5: Base size
# $6: Number of runs

# Log Info
pwd; hostname; date
# Interleave the memory pages over the NUMA domains: the examples touch their input from one thread, so all pages
# would sit in one domain (up to 10x slower at large sizes). Disable with NUMA_INTERLEAVE=0.
NUMA=""
if [[ "${NUMA_INTERLEAVE:-1}" == "1" ]] && command -v numactl >/dev/null 2>&1; then
    NUMA="numactl --interleave=all"
fi
# Pin OpenMP threads to cores (inside the CPUs Slurm gives each task)
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# Parameters
POW_START=$3
POW_STOP=$4
BASE_SIZE=$5
LOOP=$6
# Get run command
COMMAND="srun -N 1 -n 1 -c $((2**POW_START)) $NUMA"
EXECUTABLE=$1
ARGUMENTS="$BASE_SIZE $BASE_SIZE $2"
if [[ "$(basename "$EXECUTABLE")" == *"3d"* ]]; then
    ARGUMENTS="$BASE_SIZE $BASE_SIZE $BASE_SIZE $2"
fi
# Strong scaling loop from 2^pow_start to 2^pow_stop cores
$COMMAND $EXECUTABLE $((2**POW_START)) $ARGUMENTS 1 
for (( j=1; j<$LOOP; j=j+1 ))
do
    $COMMAND $EXECUTABLE $((2**POW_START)) $ARGUMENTS 0
done
for (( i=2**($POW_START+1); i<=2**$POW_STOP; i=i*2 ))
do
    COMMAND="srun -N 1 -n 1 -c $i $NUMA"
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        $COMMAND $EXECUTABLE $i $ARGUMENTS 0
    done
done
# Log Info
date

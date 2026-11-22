#!/bin/bash
#SBATCH --job-name=fftw_job          # Job name
#SBATCH --output=fftw_job.log        # Standard output
#SBATCH --error=fftw_error_%A.log    # Error Log
#SBATCH --mail-type=NONE                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --time=2:00:00                 # Time limit hrs:min:sec
#SBATCH --exclusive                     # Exclusive ressource access

# Benchmark script for shared memory problem size scaling
# $1: Executable name
# $2: FFTW planning flag (estimate/measure)
# $3: Number of cores
# $4: Starting size power
# $5: Stopping size power
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
THREADS=$3
POW_START=$4
POW_STOP=$5
LOOP=$6
# Get run command
COMMAND="srun -N 1 -n 1 -c $THREADS $NUMA"
EXECUTABLE=$1
ARGUMENTS="$((2**POW_START)) $((2**POW_START)) $2"
if [[ "$(basename "$EXECUTABLE")" == *"3d"* ]]; then
    ARGUMENTS="$((2**POW_START)) $((2**POW_START)) $((2**POW_START)) $2"
fi
# Problem size scaling loop from 2^pow_start to 2^pow_stop
$COMMAND $EXECUTABLE $THREADS $ARGUMENTS 1
for (( j=1; j<$LOOP; j=j+1 ))
do
    $COMMAND $EXECUTABLE $THREADS $ARGUMENTS 0
done
for (( i=2**($POW_START+1); i<=2**$POW_STOP; i=i*2 ))
do
    ARGUMENTS="$i $i $2"
    if [[ "$(basename "$EXECUTABLE")" == *"3d"* ]]; then
        ARGUMENTS="$i $i $i $2"
    fi
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        $COMMAND $EXECUTABLE $THREADS $ARGUMENTS 0
    done
done
# Log Info
date

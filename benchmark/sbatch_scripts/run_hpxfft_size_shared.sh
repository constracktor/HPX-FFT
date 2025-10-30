#!/bin/bash
#SBATCH --job-name=hpxfft_job          # Job name
#SBATCH --output=hpxfft_job.log        # Standard output and error log
#SBATCH --mail-type=NONE               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alexander.strack@ipvs.uni-stuttgart.de       # Where to send mail	
#SBATCH --time=2:00:00                 # Time limit hrs:min:sec
#SBATCH --exclusive                    # Exclusive ressource access

# Benchmark script for shared memory problem size scaling
# $1: Executable name
# $2: FFTW planning flag (estimate/measure)
# $3: Number of cores
# $4: Starting size power
# $5: Stopping size power
# $6: Number of runs

# Log Info
pwd; hostname; date
# Parameters
THREADS=$3
POW_START=$4
POW_STOP=$5
LOOP=$6
# Get run command
COMMAND="srun -N 1 -n 1 -c $THREADS"
EXECUTABLE=$1
ARGUMENTS="--nx=$((2**POW_START)) --ny=$((2**POW_START)) --plan=$2"
# Problem size scaling loop from 2^pow_start to 2^pow_stop
$COMMAND $EXECUTABLE $ARGUMENTS --header=true
for (( j=1; j<$LOOP; j=j+1 ))
do
    $COMMAND $EXECUTABLE $ARGUMENTS
done
for (( i=2**($POW_START+1); i<=2**$POW_STOP; i=i*2 ))
do
    ARGUMENTS="--nx=$i --ny=$i --plan=$2"
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        $COMMAND $EXECUTABLE $ARGUMENTS
    done
done
# Log Info
date

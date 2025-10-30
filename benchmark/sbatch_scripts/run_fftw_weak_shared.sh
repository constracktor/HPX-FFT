#!/bin/bash
#SBATCH --job-name=fftw_job          # Job name
#SBATCH --output=fftw_job.log        # Standard output and error log
#SBATCH --mail-type=NONE                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alexander.strack@ipvs.uni-stuttgart.de       # Where to send mail	
#SBATCH --time=2:00:00                 # Time limit hrs:min:sec
#SBATCH --exclusive                     # Exclusive ressource access

# Benchmark script for shared memory weak scaling
# $1: Executable name
# $2: FFTW planning flag (estimate/measure)
# $3: Starting number of cores
# $4: Stopping number of cores
# $5: Base size
# $6: Number of runs

# Log Info
pwd; hostname; date
# Parameters
POW_START=$3
POW_STOP=$4
BASE_SIZE=$5
LOOP=$6
# Get run command
COMMAND="srun -N 1 -n 1 -c $((2**POW_START))"
EXECUTABLE=$1
ARGUMENTS="$BASE_SIZE $BASE_SIZE $2"
# Strong scaling loop from 2^pow_start to 2^pow_stop cores
$COMMAND $EXECUTABLE $((2**POW_START)) $ARGUMENTS 1 
for (( j=1; j<$LOOP; j=j+1 ))
do
    $COMMAND $EXECUTABLE $((2**POW_START)) $ARGUMENTS 0
done
for (( i=2**($POW_START+1); i<=2**$POW_STOP; i=i*2 ))
do
    COMMAND="srun -N 1 -n 1 -c $i"
    ARGUMENTS="$((BASE_SIZE*i)) $((BASE_SIZE*i)) $2"
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        $COMMAND $EXECUTABLE $i $ARGUMENTS 0
    done
done
# Log Info
date

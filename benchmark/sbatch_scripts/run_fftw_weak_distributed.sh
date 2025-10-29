#!/bin/bash
#SBATCH --job-name=fftw_job          # Job name
#SBATCH --output=fftw_job.log        # Standard output and error log
#SBATCH --mail-type=NONE               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alexander.strack@ipvs.uni-stuttgart.de       # Where to send mail	
#SBATCH --time=2:00:00                 # Time limit hrs:min:sec
#SBATCH --exclusive                    # Exclusive ressource access

# Benchmark script for distributed memory weak scaling
# $1: Executable name
# $2: FFTW planning flag (estimate/measure)
# $3: Starting number of nodes
# $4: Stopping number of nodes
# $5: Base size
# $6: Number of threads per node
# $7: Number of runs

# Log Info
pwd; hostname; date
# Parameters
POW_START=$3
POW_STOP=$4
BASE_SIZE=$5
THREADS=$6
LOOP=$7
# Get run command
COMMAND="srun -N $((2**POW_START)) -n $((2**POW_START)) -c $THREADS"
EXECUTABLE=$1
ARGUMENTS="$BASE_SIZE $BASE_SIZE $2"
# Weak scaling loop from 2^pow_start to 2^pow_stop nodes
$COMMAND $EXECUTABLE $THREADS $ARGUMENTS 1
for (( j=1; j<$LOOP; j=j+1 ))
do
    $COMMAND $EXECUTABLE $THREADS $ARGUMENTS 0
done
for (( i=2**($POW_START+1); i<=2**$POW_STOP; i=i*2 ))
do
    COMMAND="srun -N $i -n $i -c $THREADS"
    ARGUMENTS="$((BASE_SIZE*i)) $((BASE_SIZE*i)) $2"
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        $COMMAND $EXECUTABLE $THREADS $ARGUMENTS 0
    done
done
# Log Info
date

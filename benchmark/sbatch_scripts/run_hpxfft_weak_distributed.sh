#!/bin/bash
#SBATCH --job-name=hpxfft_job          # Job name
#SBATCH --output=hpxfft_job.log        # Standard output and error log
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
# $6: HPX collective (scatter/all_to_all)
# $7: Number of threads per node
# $8: Number of runs

# Log Info
pwd; hostname; date
# Parameters
POW_START=$3
POW_STOP=$4
BASE_SIZE=$5
THREADS=$7
LOOP=$8
# Get run command
OPTIONS=""
COMMAND="srun -N $((2**POW_START)) -n $((2**POW_START)) -c $THREADS"
EXECUTABLE=$1
ARGUMENTS="--nx=$BASE_SIZE --ny=$BASE_SIZE --plan=$2 --run=$6"

# Strong scaling loop from 2^pow_start to 2^pow_stop nodes
HPX_COMMANDLINE_OPTIONS=$OPTIONS $COMMAND $EXECUTABLE $ARGUMENTS --header=true
for (( j=1; j<$LOOP; j=j+1 ))
do
    HPX_COMMANDLINE_OPTIONS=$OPTIONS $COMMAND $EXECUTABLE $ARGUMENTS
done
for (( i=2**($POW_START+1); i<=2**$POW_STOP; i=i*2 ))
do
    COMMAND="srun -N $i -n $i -c $THREADS"
    ARGUMENTS="--nx=$((BASE_SIZE*i)) --ny=$((BASE_SIZE*i)) --plan=$2"
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        HPX_COMMANDLINE_OPTIONS=$OPTIONS $COMMAND $EXECUTABLE $ARGUMENTS
    done
done
# Log Info
date

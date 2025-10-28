#!/bin/bash
#SBATCH --job-name=hpxfft_job          # Job name
#SBATCH --output=hpxfft_job.log        # Standard output and error log
#SBATCH --mail-type=NONE               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alexander.strack@ipvs.uni-stuttgart.de       # Where to send mail	
#SBATCH --time=2:00:00                 # Time limit hrs:min:sec
#SBATCH --exclusive                    # Exclusive ressource access

# Benchmark script for message size scaling between two nodes
# $1: Executable name
# $2: Base size
# $3: Stopping power of two
# $4: HPX collective (scatter/all_to_all)
# $5: Number of threads per node
# $6: Number of runs

# Log Info
pwd; hostname; date
# Parameters
BASE_SIZE=$2
POW_STOP=$3
THREADS=$5
LOOP=$6
# Get run command
OPTIONS=""
COMMAND="srun -N 2 -n 2 -c $THREADS"
EXECUTABLE=$1
ARGUMENTS="--nx=$BASE_SIZE --ny=$BASE_SIZE --plan="estimate" --run=$4"
# Message size scaling loop on 2 nodes
$COMMAND $EXECUTABLE $ARGUMENTS --header=true 
for (( j=1; j<$LOOP; j=j+1 ))
do
    $COMMAND $EXECUTABLE $ARGUMENTS
done
for (( i=2; i<=2**$POW_STOP; i=i*2 ))
do
    SIZE=$((i*BASE_SIZE))
    ARGUMENTS="--nx=$SIZE --ny=$SIZE --plan="estimate" --run=$4"
    for (( j=0; j<$LOOP; j=j+1 ))
    do
        $COMMAND $EXECUTABLE $ARGUMENTS
    done
done
# Log Info
date

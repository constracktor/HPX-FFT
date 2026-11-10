#!/bin/bash
# Run a test executable on N localities of this machine (TCP): run_localities.sh <localities> <exe> [args...]
# Fails if any locality fails or the run takes longer than 180 s.
n=$1; exe=$2; shift 2
if [ "$n" -eq 1 ]; then
    exec timeout 180 "$exe" "$@" --hpx:threads=2
fi
port=$((20000 + (RANDOM % 20000)))
pids=()
for ((i = 0; i < n; i++)); do
    timeout 180 "$exe" "$@" --hpx:localities=$n --hpx:node=$i --hpx:agas=localhost:$port \
        --hpx:hpx=localhost:$((port + 1 + i)) --hpx:threads=2 &
    pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
exit $rc

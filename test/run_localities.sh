#!/bin/bash
# Run a test executable on N localities of this machine (like hpxrun.py -l N): run_localities.sh <localities> <exe> [args...]
# Uses the default HPX ports, so tests that use it must not run concurrently. Fails if any locality fails, if the
# number of PASS lines differs from the number of localities, or if the run takes longer than 180 s.
n=$1; exe=$2; shift 2
out=$(mktemp)
trap 'rm -f "$out"' EXIT
pids=()
if [ "$n" -eq 1 ]; then
    timeout 180 "$exe" "$@" --hpx:threads=2 --hpx:ignore-batch-env >>"$out" 2>&1 &
    pids+=($!)
else
    for ((i = 0; i < n; i++)); do
        timeout 180 "$exe" "$@" --hpx:localities=$n --hpx:node=$i --hpx:threads=2 --hpx:ignore-batch-env >>"$out" 2>&1 &
        pids+=($!)
    done
fi
rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
cat "$out"
passes=$(grep -c '^PASS' "$out")
if [ "$passes" -ne "$n" ]; then
    echo "run_localities: expected $n PASS lines, found $passes"
    rc=1
fi
exit $rc

#!/bin/bash

for d in */; do
    folder="${d%/}"
    outfile="$folder/$folder.out"

    if [[ -f "$outfile" ]]; then
        energy=$(grep "OPT END - CONVERGED" "$outfile" | awk '{print $8}')
        printf "%-30s %s\n" "$folder" "$energy"
    else
        printf "%-30s %s\n" "$folder" "NO_OUTFILE"
    fi
done
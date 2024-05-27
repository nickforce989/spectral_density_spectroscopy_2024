#!/bin/bash

subdirs=("M1" "M2" "M3" "M4" "M5")

cd lsd_out

# Check if LaTeX is installed
if ! command -v latex > /dev/null 2>&1; then
    echo "LaTeX is not installed. Producing CSVs, no plots nor tables.tex files"
    inner_condition_met=true
    for subdir in "${subdirs[@]}"; do
        echo "Checking subdirectories for pattern: ${subdir}_*"
        count=$(find . -maxdepth 1 -type d -name "${subdir}_*" | wc -l)
        echo "Found $count subdirectories matching ${subdir}_*"
        if [ "$count" -ne 24 ]; then
            inner_condition_met=false
            break
        fi
    done

    cd ..
    if [ "$inner_condition_met" = false ]; then
        bash run_plateaus.sh
        bash run_spectral_densities.sh
    else
        bash run_spectral_densities.sh
    fi

    
else
    echo "Latex is installed. Running full workflow."
    inner_condition_met=true
    for subdir in "${subdirs[@]}"; do
        echo "Checking subdirectories for pattern: ${subdir}_*"
        count=$(find . -maxdepth 1 -type d -name "${subdir}_*" | wc -l)
        echo "Found $count subdirectories matching ${subdir}_*"
        if [ "$count" -ne 24 ]; then
            inner_condition_met=false
            break
        fi
    done
    cd ..

    if [ "$inner_condition_met" = false ]; then
	bash run_plateaus.sh
        bash run_spectral_densities.sh
        python CSVs_to_tables.py
        bash run_plots.sh
    else
        bash run_spectral_densities.sh
        python CSVs_to_tables.py
        bash run_plots.sh
    fi
fi

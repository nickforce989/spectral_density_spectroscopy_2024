#!/bin/bash


# Check if LaTeX is installed
if ! command -v latex > /dev/null 2>&1; then
    echo "LaTeX is not installed. Producing CSVs, no plots nor tables.tex files"
    #bash run_plateaus.sh
    bash run_spectral_densities.sh
    
else
    echo "Latex is installed. Running full workflow."
    #bash run_plateaus.sh
    bash run_spectral_densities.sh
    python CSVs_to_tables.py
    bash run_plots.sh
fi

#!/bin/bash

bash check_latex.sh
bash run_plateaus.sh
bash run_spectral_densities.sh
python CSVs_to_tables.py
bash run_plots.sh

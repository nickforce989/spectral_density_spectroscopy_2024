#!/bin/bash

cd just_plotting/code
cd topologies
bash main.sh
cd ../improving_spectrum
python improvement_spectrum_nt.py
cd ../kernel_comparisons_nt
python plot_improvement_nt.py
cd ../kernel_worsening
python plot_worsening.py
cd ../sigma_variation
python plot_sigma_variation.py
cd ../stability_plot
python stability.py
cd ../systematic_errors
python plot_spectrum.py
python plot_spectrum2.py
python plot_spectrum3.py
python plot_spectrum4.py
cd ../two_kernels
python different_kernels.py
cd ../final_spectrum
python spectrum_MN.py
cd ../../../plateaus
python plot_GEVP_F.py
python plot_GEVP_AS.py


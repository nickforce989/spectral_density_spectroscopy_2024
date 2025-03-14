#!/bin/bash

bash check_latex.sh

cd just_plotting/code
cd topologies
bash main.sh

# Check if the 'input_fit' directory is not empty
if [ "$(ls -A ../../../input_fit | wc -l)" -gt 1 ]; then
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
    python spectrum_PS.py
    cd ../Fig7
    python check_meson.py
    python fit_figure_gauss.py
    python fit_figure_cauchy.py
    cd ../Fig10
    python fit_figure_gauss.py
    cd ../Fig11
    python fit_figure_gauss_up.py
    python fit_figure_gauss_down.py
    cd ../Fig12
    python fit_figure_gauss.py
    cd ../Fig13
    python fit_figure_gauss.py
    cd ../Fig15
    python fit_figure_gauss_up.py
    python fit_figure_gauss_middle.py
    python fit_figure_gauss_down.py
fi

cd ../../../plateaus
python plot_GEVP_F.py
python plot_GEVP_AS.py


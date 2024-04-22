# Analysis code for (arXiv: ######)

This is the analysis code built on top of **lsdensities** code (
<a href="https://github.com/LupoA/lsdensities"> GitHub repository </a>) to
reproduce the results in **(arXiv: ######)**.

## Authors

Niccolò Forzano, Ho Hsiao, Fabian Zierler

## Usage

To reproduce all the plots that are shown in the papers, deriving from 
spectral density findings and GEVP, run ``bash run_plots.sh``. To do so,
ensure that the CSVs in 'plateaus/CSVs' are full.

To find all the spectral densities that have been used in the paper,
ensure that the HDF5 file containing all the data is present in the 
directory ``input_correlators/``, and then run ``bash run_spectral_densities.sh``.

Spectral density fits can be reproduced by using the example code in LSDensities 
``lsd_out/lsdensities/examples/runFitRho.py``.

Single plateaus fits can be reproduced by using the example code
in ``plateaus/check_meson.py``.

To find all the GEVP plateaus results that are shown in the papers,
run ``bash run_plateaus.sh``. Firstly, ensure that the HDF5 file containing 
all the data is present in the  directory ``input_correlators/``.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[GPL](https://choosealicense.com/licenses/gpl-3.0/)

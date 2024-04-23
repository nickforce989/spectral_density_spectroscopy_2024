# Analysis code for (arXiv: ######)

This is the analysis code built on top of **lsdensities** code (
<a href="https://github.com/LupoA/lsdensities"> GitHub repository </a>) to
reproduce the results in **(arXiv: ######)**.

## Authors

Niccolò Forzano, Ho Hsiao, Fabian Zierler


## Set up environment

Download this code and download the data release files in doi:######.zenodo.#########

Then, create the conda environment in terminal with conda installed:

```
conda env create -f environment.yml
```

Once the environment is created, you can active it:

```
conda activate analysis-env
```

## Code usage

To reproduce all the plots that are shown in the papers, deriving from 
spectral density findings and GEVP (Figs. 1, 2, 3, 4, 5, 6, 8, 9, 14, 16, 17, 18, 19, 20), run ``bash run_plots.sh``. To do so,
ensure that the CSVs in ``plateaus/CSVs`` are full.

To find all the spectral densities from scratch and exactly how they have been used in the paper,
ensure that the HDF5 file containing all the data is present in the 
directory ``input_correlators/``, and then run ``bash run_spectral_densities.sh``.

Spectral density fits (Figs. 7, 15, 10 right panel, 11 right panel, 12 right panel, 13 right panel) can be reproduced by using the example code in LSDensities 
``lsd_out/lsdensities/examples/runFitRho.py``.

Single plateaus fits (Figs. 10 left panel, 11 left panel, 12 left panel, 13 left panel) can be reproduced by using the example code
in ``plateaus/check_meson.py``.

To find all the GEVP plateaus results from scratch, selecting the plateaus
extents by hand, run ``bash run_plateaus.sh``. Firstly, ensure that the HDF5
file containing all the data is present in the  directory ``input_correlators/``.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[GPL](https://choosealicense.com/licenses/gpl-3.0/)

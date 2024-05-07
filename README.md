# Analysis code for (arXiv:2405.01388)

This is the analysis code built on top of **lsdensities** code (
<a href="https://github.com/LupoA/lsdensities"> GitHub repository </a>) to
reproduce the results in **(arXiv:2405.01388)**.

## Authors

Niccolò Forzano, Ho Hsiao, Fabian Zierler


## Set up environment

Download this code and download the data release files in doi:10.5281/zenodo.11048346.
From there, put ``chimera_data_full.hdf5`` in ``input_correlators/`` and the content of 
``input_topology.zip`` in ``input_topology/``.

Then, create the conda environment in terminal with conda installed:

```
conda env create -f environment.yml
```
with the caveat that if you're using an Apple silicon CPU then you need to use Conda 24.3 or later, and specify ```--platform osx-64```
in your ```conda env create``` call.

Once the environment is created, you can active it:

```
conda activate analysis-env
```

## Code usage

To reproduce all the plots that are shown in the papers, run 
``bash run_plots.sh``. To do so, ensure that the CSVs in ``plateaus/CSVs`` are full. Please, also ensure that ``input_topology/`` is full.

To find all the spectral densities from scratch and exactly how they have been used in the paper,
ensure that the HDF5 file containing all the data is present in the 
directory ``input_correlators/``, and then run ``bash run_spectral_densities.sh``.

Spectral density fits can be reproduced by using ``lsd_out/fit_data.py``.

To find all the GEVP plateaus results from scratch, selecting the plateaus
extents by hand, run ``bash self_finding_plateaus.sh``. Firstly, ensure that the HDF5
file containing all the data is present in the  directory ``input_correlators/``.
The command ``bash run_plateaus.sh``, instead, reproduces the plateaus by using the 
metadata used for the analysis in the paper. 

## Acknoledgement

The flow_analysis code in ```topologies/flow_analysis``` has been based on the following <a href="https://github.com/edbennett/flow_analysis/"> GitHub repository </a>.

## License

[GPL](https://choosealicense.com/licenses/gpl-3.0/)

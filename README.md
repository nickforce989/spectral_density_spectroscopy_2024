# Analysis code for (arXiv:2405.01388)

This is the analysis code built on top of **lsdensities** code (
<a href="https://github.com/LupoA/lsdensities"> GitHub repository </a>) to
reproduce the results in **(arXiv:2405.01388)**.

## Authors

Niccolò Forzano, Ho Hsiao, Fabian Zierler, Ed Bennett, Luigi Del Debbio, Ryan C. Hill,
Deog Ki Hong, Jong-Wan Lee, C.-J. David Lin, Biagio Lucini, Alessandro Lupo,
Maurizio Piai, Davide Vadacchino.


## Set up environment

* Download this code and download the data release files in doi:10.5281/zenodo.11048346.
  From there, put ``chimera_data_full.hdf5`` in ``input_correlators/`` and the content of 
  ``input_topology.zip`` in ``input_topology/``.


* Then, create the conda environment in terminal with conda installed:

```
conda env create -f environment.yml
```
  
  with the caveat that if you're using an Apple silicon CPU then you need to use Conda 24.3 or later, and specify ```--platform osx-64```
  in your ```conda env create``` call.


* Once the environment is created, you can active the it:

```
conda activate analysis-env
```

## Code usage

* To reproduce all the plots that are shown in the paper, run 
  ``bash run_plots.sh``. To do so, ensure that the CSVs in ``plateaus/CSVs`` are full. 
  Please, make sure that also ``input_topology/`` is full. The results will be found in
  ``plots/``.

* To find all the spectral densities from scratch, and fit them run ``run_spectral_densities.sh``. 
  While running this file:
   * If ``input_fit/`` has been filled (using the corresponding directory in doi:10.5281/zenodo.11048346)
     the fitting procedure will be applied to pre-reconstructed spectral densities.
   * If ``input_fit/`` is empty, the code will reconstruct from scratch the spectral densities and then
     to fit them. This procedure may take quite long time.

  Please, make sure that the HDF5 file containing all the data is present in the 
  directory ``input_correlators/`` before running ``bash run_spectral_densities.sh``.

* The command ``bash run_plateaus.sh`` reproduces the GEVP plateaus by using the 
  metadata used for the analysis in the paper. Firstly, make sure that the HDF5
  file containing all the data is present in the  directory ``input_correlators/``.

* To reproduce all the plots and results in the tables present in the paper, please run
  ``reproduce_everything.sh``. Please, make sure that ``input_fit/``, ``input_topology/`` and
  ``input_correlators/`` are full, before running. The results will be found in ``plots/`` and
  ``tables/``.

## Acknoledgement

The flow_analysis code in ```topologies/flow_analysis``` has been based on the following <a href="https://github.com/edbennett/flow_analysis/"> GitHub repository </a>.

## License

[GPL](https://choosealicense.com/licenses/gpl-3.0/)

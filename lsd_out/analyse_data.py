import read_hdf
import translate
import numpy as np
import csv
import multiprocessing
import lsdensities.utils.rhoUtils as u
from lsdensities.utils.rhoUtils import (
    init_precision,
    LogMessage,
    end,
    Inputs,
    generate_seed,
)
from lsdensities.utils.rhoParser import parseArgumentRhoFromData
from lsdensities.utils.rhoUtils import create_out_paths
from lsdensities.correlator.correlatorUtils import symmetrisePeriodicCorrelator
from lsdensities.utils.rhoParallelUtils import ParallelBootstrapLoop
import os
from mpmath import mp, mpf
import numpy as np
from lsdensities.InverseProblemWrapper import AlgorithmParameters, InverseProblemWrapper
from lsdensities.utils.rhoUtils import MatrixBundle
import random
def main():
    def init_variables(datapath, outdir, ne, emin, emax, periodicity, kernel, sigma, prec, nboot, e0, Na, A0cut, mpi):
        in_ = Inputs()
        in_.tmax = 0
        in_.periodicity = periodicity
        in_.kerneltype = kernel
        in_.prec = prec
        in_.datapath = datapath
        in_.outdir = outdir
        in_.massNorm = mpi
        in_.num_boot = nboot
        in_.sigma = sigma
        in_.emax = (
            emax * mpi
        )  #   we pass it in unit of Mpi, here to turn it into lattice (working) units
        if emin == 0:
            in_.emin = (mpi / 20) * mpi
        else:
            in_.emin = emin * mpi
        in_.e0 = e0
        in_.Ne = ne
        in_.Na = Na
        in_.A0cut = A0cut
        return in_
    def findRho(datapath, outdir, ne, emin, emax, periodicity, kernel, sigma, prec, nboot, e0, Na, A0cut, mpi):
        print(LogMessage(), "Initialising")
        #args = parseArgumentRhoFromData()
        init_precision(prec)
        par = init_variables(datapath, outdir, ne, emin, emax, periodicity, kernel, sigma, prec, nboot, e0, Na, A0cut, mpi)

        seed = generate_seed(par)
        random.seed(seed)
        np.random.seed(random.randint(0, 2 ** (32) - 1))

        #   Reading datafile, storing correlator
        rawcorr, par.time_extent, par.num_samples = u.read_datafile(par.datapath)
        par.tmax = int(par.time_extent / 2)
        par.assign_values()
        par.report()
        par.plotpath, par.logpath = create_out_paths(par)

        #   Correlator
        rawcorr.evaluate()
        rawcorr.tmax = par.tmax
        if par.periodicity == "COSH":
            print(LogMessage(), "Folding correlator")
            symCorr = symmetrisePeriodicCorrelator(corr=rawcorr, par=par)
            symCorr.evaluate()

        #   Resampling
        if par.periodicity == "EXP":
            corr = u.Obs(
                T=par.time_extent, tmax=par.tmax, nms=par.num_boot, is_resampled=True
            )
            resample = ParallelBootstrapLoop(par, rawcorr.sample, is_folded=False)
        if par.periodicity == "COSH":
            corr = u.Obs(
                T=symCorr.T,
                tmax=symCorr.tmax,
                nms=par.num_boot,
                is_resampled=True,
            )
            resample = ParallelBootstrapLoop(par, symCorr.sample, is_folded=False)

        corr.sample = resample.run()
        corr.evaluate()
        #   -   -   -   -   -   -   -   -   -   -   -

        #   Covariance
        print(LogMessage(), "Evaluate covariance")
        corr.evaluate_covmatrix(plot=False)
        corr.corrmat_from_covmat(plot=False)
        with open(os.path.join(par.logpath, "covarianceMatrix.txt"), "w") as output:
            for i in range(par.time_extent):
                for j in range(par.time_extent):
                    print(i, j, corr.cov[i, j], file=output)
        #   -   -   -   -   -   -   -   -   -   -   -

        #   Turn correlator into mpmath variable
        print(LogMessage(), "Converting correlator into mpmath type")
        corr.fill_mp_sample()
        print(LogMessage(), "Cond[Cov C] = {:3.3e}".format(float(mp.cond(corr.mpcov))))

        #   Prepare
        cNorm = mpf(str(corr.central[1] ** 2))
        lambdaMax = 1e4
        energies = np.linspace(par.emin, par.emax, par.Ne)

        hltParams = AlgorithmParameters(
            alphaA=0,
            alphaB=1 / 2,
            alphaC=+1.99,
            lambdaMax=lambdaMax,
            lambdaStep=lambdaMax / 2,
            lambdaScanCap=8,
            kfactor=0.1,
            lambdaMin=5e-2,
            comparisonRatio=0.3,
        )
        matrix_bundle = MatrixBundle(Bmatrix=corr.mpcov, bnorm=cNorm)

        HLT = InverseProblemWrapper(
            par=par,
            algorithmPar=hltParams,
            matrix_bundle=matrix_bundle,
            correlator=corr,
            energies=energies,
        )
        HLT.prepareHLT()
        HLT.run()
        HLT.stabilityPlot(
            generateHLTscan=True,
            generateLikelihoodShared=True,
            generateLikelihoodPlot=True,
            generateKernelsPlot=True,
        )  # Lots of plots as it is
        HLT.plotResult()
        end()

    ####################### External data for make rho finding easier #######################
    categories = ['PS', 'V', 'T', 'AV', 'AT', 'S', 'ps', 'v', 't', 'av', 'at', 's']
    # Mesonic channels
    mesonic_channels = ['g5', 'gi', 'g0gi', 'g5gi', 'g0g5gi', 'id']
    # Ensembles: M1, M2, M3, M4, M5
    ensembles = ['M1', 'M2', 'M3', 'M4', 'M5']
    # Roots in HDF5 for each ensemble
    roots = ['chimera_out_48x20x20x20nc4nf2nas3b6.5mf0.71mas1.01_APE0.4N50_smf0.2as0.12_s1',
             'chimera_out_64x20x20x20nc4nf2nas3b6.5mf0.71mas1.01_APE0.4N50_smf0.2as0.12_s1',
             'chimera_out_96x20x20x20nc4nf2nas3b6.5mf0.71mas1.01_APE0.4N50_smf0.2as0.12_s1',
             'chimera_out_64x20x20x20nc4nf2nas3b6.5mf0.70mas1.01_APE0.4N50_smf0.2as0.12_s1',
             'chimera_out_64x32x32x32nc4nf2nas3b6.5mf0.72mas1.01_APE0.4N50_smf0.24as0.12_s1']
    # Representations considered
    reps = ['fund', 'anti']
    # Kernel in HLT
    kerneltype = ['HALFNORMGAUSS', 'CAUCHY']
    # Initialize dictionaries to store the data
    Nsource_C_values_MN = {}
    Nsink_C_values_MN = {}
    am_C_values_MN = {}
    sigma1_over_mC_values_MN = {}
    sigma2_over_mC_values_MN = {}
    # Read data from CSV
    with open('metadata/metadata_spectralDensity.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ensemble = row['Ensemble']
            # Initialize lists for each ensemble if not already present
            if ensemble not in Nsource_C_values_MN:
                Nsource_C_values_MN[ensemble] = []
                Nsink_C_values_MN[ensemble] = []
                am_C_values_MN[ensemble] = []
                sigma1_over_mC_values_MN[ensemble] = []
                sigma2_over_mC_values_MN[ensemble] = []
            # Append data for each category to the respective lists
            for category in categories:
                Nsource_C_values_MN[ensemble].append(int(row[f'{category}_Nsource']))
                Nsink_C_values_MN[ensemble].append(int(row[f'{category}_Nsink']))
                am_C_values_MN[ensemble].append(float(row[f'{category}_am']))
                sigma1_over_mC_values_MN[ensemble].append(float(row[f'{category}_sigma1_over_m']))
                sigma2_over_mC_values_MN[ensemble].append(float(row[f'{category}_sigma2_over_m']))
    # Create a 3D matrix with ensemble index
    matrix_4D = [
        [
            ensemble,
            am_C_values_MN[ensemble],
            sigma1_over_mC_values_MN[ensemble],
            sigma2_over_mC_values_MN[ensemble],
            Nsource_C_values_MN[ensemble],
            Nsink_C_values_MN[ensemble]
        ]
        for ensemble in ensembles
    ]
    def process_channel(channel, k, index, rep, ensemble, kernel, matrix_4D, roots, file_path):
        Nsource = matrix_4D[index][4][k]
        Nsink = matrix_4D[index][5][k]
        group_prefixes = {
            'gi': ['g1', 'g2', 'g3'],
            'g0gi': ['g0g1', 'g0g2', 'g0g3'],
            'g5gi': ['g5g1', 'g5g2', 'g5g3'],
            'g0g5gi': ['g0g5g1', 'g0g5g2', 'g0g5g3']
        }
        prefix = group_prefixes.get(channel, [channel])
        datasets = []
        for g in prefix:
            dataset_path = f"{roots[index]}/source_N{Nsource}_sink_N{Nsink}/{rep} TRIPLET {g}"
            group1 = f"source_N{Nsource}_sink_N{Nsink}"
            group2 = f"{rep} TRIPLET {g}"
            datasets.append(read_hdf.extract_dataset(file_path, group2, roots[index], group1))
            with open('paths.log', 'a') as file:
                print(dataset_path, file=file)
        dataset = sum(datasets) / len(datasets) if datasets else None
        if channel == 'id' or channel == 'g5':
            group2 = f"{rep} TRIPLET {channel}"
            dataset_path = f"{roots[index]}/{channel}/{rep} TRIPLET {channel}"
            group1 = f'source_N{Nsource}_sink_N{Nsink}'
            dataset = read_hdf.extract_dataset(file_path, group2, roots[index], group1)
            with open('paths.log', 'a') as file:
                print(dataset_path, file=file)
        if dataset is not None:
            translate.save_matrix_to_file(dataset, f'corr_to_analyse_{channel}_{rep}_{ensemble}.txt')
        mpi = matrix_4D[index][1][k]
        if kernel == 'HALFNORMGAUSS':
            if rep == 'fund':
                tmp = mpi * matrix_4D[index][2][k]
            else:
                tmp = mpi * matrix_4D[index][2][k + 6]
        elif kernel == 'CAUCHY':
            if rep == 'fund':
                tmp = mpi * matrix_4D[index][3][k]
            else:
                tmp = mpi * matrix_4D[index][3][k + 6]
        mpi = mpi
        sigma = tmp
        decimal_part = tmp / matrix_4D[index][1][k] % 1
        decimal_as_int = int(decimal_part * 100)
        datapath = f'./corr_to_analyse_{channel}_{rep}_{ensemble}.txt'
        outdir = f'./{ensemble}_{rep}_{channel}_s0p{decimal_as_int}_{kernel}_Nsource{Nsource}_Nsink{Nsink}'
        ne = 1
        emin = 0.3
        emax = 2.2
        periodicity = 'COSH'
        prec = 105
        nboot = 300
        e0 = 0.0
        Na = 1
        A0cut = 0.1
        findRho(datapath, outdir, ne, emin, emax, periodicity, kernel, sigma, prec, nboot, e0, Na, A0cut, mpi)
    ################# Download and use lsdensities on correlators ########################
    # Replace 'your_file.h5' with the path to your HDF5 file
    file_path = '../input_correlators/chimera_data_full.hdf5'
    for kernel in kerneltype:
        for index, ensemble in enumerate(ensembles):
            for rep in reps:
                processes = []
                for k, channel in enumerate(mesonic_channels):
                    process = multiprocessing.Process(target=process_channel, args=(
                    channel, k, index, rep, ensemble, kernel, matrix_4D, roots, file_path))
                    processes.append(process)
                    process.start()
                for process in processes:
                    process.join()

if __name__ == "__main__":
    main()
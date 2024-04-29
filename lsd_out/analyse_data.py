import subprocess
import read_hdf
import translate
import numpy as np
import csv
import multiprocessing
from runInverseProblem import init_variables, findRho
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

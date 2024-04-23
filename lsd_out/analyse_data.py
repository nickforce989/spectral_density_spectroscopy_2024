import subprocess
import read_hdf
import translate
import numpy as np
import csv

####################### External data for make rho finding easier #######################

categories = ['PS', 'V', 'T', 'AV', 'AT', 'S', 'ps', 'v', 't', 'av', 'at', 's']

# Mesonic channels
mesonic_channels = ['g5', 'g1', 'g0g1', 'g5g1', 'g0g5g1', 'id']

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

################# Download and use lsdensities on correlators ########################

# Clone the repository
subprocess.run(['git', 'clone', 'https://github.com/LupoA/lsdensities.git'])

# Change into the cloned directory
subprocess.run(['cd', 'lsdensities'], shell=True)

# Install the required dependencies
subprocess.run(['pip', 'install', '-r', 'lsdensities/requirements.txt'])

# Replace 'your_file.h5' with the path to your HDF5 file
file_path = '../input_correlators/chimera_data_full.hdf5'

kerneltype = ['HALFNORMGAUSS', 'CAUCHY']

for kernel in kerneltype:
    for index, ensemble in enumerate(ensembles):
        for rep in reps:
            for k, channel in enumerate(mesonic_channels):
            #for k in range(1):
                Nsource = matrix_4D[index][4][k]
                Nsink = matrix_4D[index][5][k]
                ####################### Get correlators from HDF5 file #################################
                if channel == 'g1':
                    dataset_path1 =  roots[index] + '/' + f'source_N{Nsource}_sink_N{Nsink}/{rep} TRIPLET g1'
                    dataset_path2 =  roots[index] + '/' + f'source_N{Nsource}_sink_N{Nsink}/{rep} TRIPLET g2'
                    dataset_path3 =  roots[index] + '/' +  f'source_N{Nsource}_sink_N{Nsink}/{rep} TRIPLET g3'
                    group1_1 = f'source_N{Nsource}_sink_N{Nsink}'
                    group2_1 = f'{rep} TRIPLET g1'
                    group1_2 = f'source_N{Nsource}_sink_N{Nsink}'
                    group2_2 = f'{rep} TRIPLET g2'
                    group1_3 = f'source_N{Nsource}_sink_N{Nsink}'
                    group2_3 = f'{rep} TRIPLET g3' 

                    dataset1 = read_hdf.extract_dataset(file_path, group2_1, roots[index], group1_1)
                    dataset2 = read_hdf.extract_dataset(file_path, group2_2, roots[index], group1_2)
                    dataset3 = read_hdf.extract_dataset(file_path, group2_3, roots[index], group1_3)

                    with open('paths.log', 'a') as file:
                        print(dataset_path1, file=file)
                    dataset = (dataset1 + dataset2 + dataset3) / 3
                    channel = 'gi'
                    
                    
                elif channel == 'g0g1':
                    dataset_path1 =  roots[index] + '/' + f'source_N{Nsource}_sink_N{Nsink}/{rep} TRIPLET g0g1'
                    dataset_path2 =  roots[index] + '/' + f'source_N{Nsource}_sink_N{Nsink}/{rep} TRIPLET g0g2'
                    dataset_path3 =  roots[index] + '/' +  f'source_N{Nsource}_sink_N{Nsink}/{rep} TRIPLET g0g3'
                    group1_1 = f'source_N{Nsource}_sink_N{Nsink}'
                    group2_1 = f'{rep} TRIPLET g0g1'
                    group1_2 = f'source_N{Nsource}_sink_N{Nsink}'
                    group2_2 = f'{rep} TRIPLET g0g2'
                    group1_3 = f'source_N{Nsource}_sink_N{Nsink}'
                    group2_3 = f'{rep} TRIPLET g0g3' 

                    dataset1 = read_hdf.extract_dataset(file_path, group2_1, roots[index], group1_1)
                    dataset2 = read_hdf.extract_dataset(file_path, group2_2, roots[index], group1_2)
                    dataset3 = read_hdf.extract_dataset(file_path, group2_3, roots[index], group1_3)

                    with open('paths.log', 'a') as file:
                        print(dataset_path1, file=file)
                    dataset = (dataset1 + dataset2 + dataset3) / 3
                    channel = 'g0gi'
                    
                    
                elif channel == 'g5g1':                    
                    dataset_path1 =  roots[index] + '/' + f'source_N{Nsource}_sink_N{Nsink}/{rep} TRIPLET g5g1'
                    dataset_path2 =  roots[index] + '/' + f'source_N{Nsource}_sink_N{Nsink}/{rep} TRIPLET g5g2'
                    dataset_path3 =  roots[index] + '/' +  f'source_N{Nsource}_sink_N{Nsink}/{rep} TRIPLET g5g3'
                    group1_1 = f'source_N{Nsource}_sink_N{Nsink}'
                    group2_1 = f'{rep} TRIPLET g5g1'
                    group1_2 = f'source_N{Nsource}_sink_N{Nsink}'
                    group2_2 = f'{rep} TRIPLET g5g2'
                    group1_3 = f'source_N{Nsource}_sink_N{Nsink}'
                    group2_3 = f'{rep} TRIPLET g5g3' 

                    dataset1 = read_hdf.extract_dataset(file_path, group2_1, roots[index], group1_1)
                    dataset2 = read_hdf.extract_dataset(file_path, group2_2, roots[index], group1_2)
                    dataset3 = read_hdf.extract_dataset(file_path, group2_3, roots[index], group1_3)

                    with open('paths.log', 'a') as file:
                        print(dataset_path1, file=file)
                    dataset = (dataset1 + dataset2 + dataset3) / 3
                    channel = 'g5gi'
                    
                    
                    
                elif channel == 'g0g5g1':
                
                    dataset_path1 =  roots[index] + '/' + f'source_N{Nsource}_sink_N{Nsink}/{rep} TRIPLET g0g5g1'
                    dataset_path2 =  roots[index] + '/' + f'source_N{Nsource}_sink_N{Nsink}/{rep} TRIPLET g0g5g2'
                    dataset_path3 =  roots[index] + '/' +  f'source_N{Nsource}_sink_N{Nsink}/{rep} TRIPLET g0g5g3'
                    group1_1 = f'source_N{Nsource}_sink_N{Nsink}'
                    group2_1 = f'{rep} TRIPLET g0g5g1'
                    group1_2 = f'source_N{Nsource}_sink_N{Nsink}'
                    group2_2 = f'{rep} TRIPLET g0g5g2'
                    group1_3 = f'source_N{Nsource}_sink_N{Nsink}'
                    group2_3 = f'{rep} TRIPLET g0g5g3' 

                    dataset1 = read_hdf.extract_dataset(file_path, group2_1, roots[index], group1_1)
                    dataset2 = read_hdf.extract_dataset(file_path, group2_2, roots[index], group1_2)
                    dataset3 = read_hdf.extract_dataset(file_path, group2_3, roots[index], group1_3)

                    with open('paths.log', 'a') as file:
                        print(dataset_path1, file=file)
                    dataset = (dataset1 + dataset2 + dataset3) / 3
                    channel = 'g0g5gi'

                elif channel == 'id':                  
                    dataset_path =  roots[index] + '/' + f'source_N{Nsource}_sink_N{Nsink}/{rep} TRIPLET id'
                    group1 = f'source_N{Nsource}_sink_N{Nsink}'
                    group2 = f'{rep} TRIPLET id'

                    dataset = read_hdf.extract_dataset(file_path, group2, roots[index], group1)
                    with open('paths.log', 'a') as file:
                        print(dataset_path, file=file)
                        
                elif channel == 'g5':
                    dataset_path =  roots[index] + '/' + f'source_N{Nsource}_sink_N{Nsink}/{rep} TRIPLET g5'
                    group1 = f'source_N{Nsource}_sink_N{Nsink}'
                    group2 = f'{rep} TRIPLET g5'

                    dataset = read_hdf.extract_dataset(file_path, group2, roots[index], group1)
                    with open('paths.log', 'a') as file:
                        print(dataset_path, file=file)


                # Save the data to a .txt file using the new function
                translate.save_matrix_to_file(dataset, 'corr_to_analyse.txt')
                ########################################################################################

                # Matrix_4D: get ensemble, get element (e.g mpis) for the ensemble, get subvalue of the element
                # print(matrix_4D[0][4])
                mpi = matrix_4D[index][1][k]
                if kernel == 'HALFNORMGAUSS':
                    if rep == 'fund':
                        tmp = mpi * matrix_4D[index][2][k]
                    else:
                        tmp = mpi * matrix_4D[index][2][k+6]
                elif kernel == 'CAUCHY':
                    if rep == 'fund':
                        tmp = mpi * matrix_4D[index][3][k]
                    else:
                        tmp = mpi * matrix_4D[index][3][k + 6]
                mpi = str(mpi)
                sigma = str(tmp)

                # Extract the decimal part using the modulo operator
                decimal_part = tmp / matrix_4D[index][1][k] % 1
                # Convert the decimal part to an integer
                decimal_as_int = int(decimal_part * 100)  # Multiply by 100 to get two decimal places

                # Define variables for subprocess arguments
                datapath = './corr_to_analyse.txt'
                # Your code here, where 'kernel' represents the current element in the loop
                outdir = f'./{ensemble}_{rep}_{channel}_s0p{decimal_as_int}_{kernel}_Nsource{Nsource}_Nsink{Nsink}'

                ne = '1'
                emin = '0.3'
                emax = '2.2'
                periodicity = 'COSH'

                # Run the runInverseProblem.py script with custom arguments using variables
                subprocess.run(['python', 'lsdensities/examples/runInverseProblem.py',
                                '-datapath', datapath,
                                '--outdir', outdir,
                                '--kerneltype', kernel,
                                '--mpi', mpi,
                                '--sigma', sigma,
                                '--ne', ne,
                                '--emin', emin,
                                '--emax', emax,
                                '--periodicity', periodicity])


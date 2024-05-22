import pandas as pd
import numpy as np

def add_error(channel_E0, err):
    if channel_E0 != 0 and not np.isnan(channel_E0):
        #print(err)
        # Convert the error to a string with significant digits
        err_str = f"{err:.2g}".replace('.', '')  # Convert error to string with 2 significant digits and remove decimal
        #print(err_str)
        # Count leading zeros
        leading_zeros = len(err_str) - len(err_str.lstrip('0'))

        # Remove leading zeros
        err_str = err_str.lstrip('0')

        #print(err_str)
        if len(str(err_str)) == 1:
            err_str = str(int(err_str)*10)
        # Determine the number of significant digits in the error
        err_significant_digits = len(err_str) + leading_zeros


        # Format the channel_E0 value with the correct number of significant digits
        # Calculate the format precision for the channel_E0 value
        int_part_length = len(str(int(channel_E0)).replace('.', ''))
        format_precision = max(err_significant_digits - int_part_length, 0)
        format_str = f"{{:.{format_precision}f}}"
        channel_E0_str = format_str.format(channel_E0)

        #print(len(str(err_str)))
        #print(err_str)


        # Combine the channel_E0 value with the error part
        channel_E0_with_error = f"{channel_E0_str}({err_str})"
    else:
        channel_E0_with_error = '-'
    return channel_E0_with_error



# Read CSV files
metadata = pd.read_csv('./lsd_out/metadata/metadata_spectralDensity.csv')
f_meson_gevp = pd.read_csv('./CSVs/F_meson_GEVP.csv')
as_meson_gevp = pd.read_csv('./CSVs/AS_meson_GEVP.csv')
f_mix_meson_gevp = pd.read_csv('./CSVs/F_meson_GEVP_mix.csv')
as_mix_meson_gevp = pd.read_csv('./CSVs/AS_meson_GEVP_mix.csv')

ensembles = ['M1', 'M2', 'M3', 'M4', 'M5']
prefix = ['48x20x20x20b6.5mf0.71mas1.01', '64x20x20x20b6.5mf0.71mas1.01', '96x20x20x20b6.5mf0.71mas1.01',
          '64x20x20x20b6.5mf0.70mas1.01', '64x32x32x32b6.5mf0.72mas1.01']

# Iterate through chunks of 4 rows in M3_spectral_density_spectrum.csv
chunk_size = 4

for n in range(3):
    for index, ensemble in enumerate(ensembles):
        # Initialize LaTeX table string
        latex_table = "\\begin{table}[ht]\n"
        latex_table += "\\centering\n"
        latex_table += "\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}\n"
        latex_table += "\\hline\n"
        latex_table += "$C$ & $k$ & $N_{\\text{source}}$ & $N_{\\text{sink}}$ & "f"$aE_{n}$ $k$-G & $aE_{n}$ $(k+1)$-G & $aE_{n}$ $k$-C & $aE_{n}$ $(k+1)$-C$ & am_C & ""$\sigma_{G} / m_C$ & $\sigma_{C} / m_C$ \\\\\n"
        latex_table += "\\hline\n"
        for chunk in pd.read_csv(f'./CSVs/{ensemble}_spectral_density_spectrum.csv', chunksize=chunk_size):
            channel = chunk['channel'].min()
            repr = chunk['rep'].min()
            print(channel)
            print(repr)
            if channel == 'g5' and repr == 'fund':
                CHANNEL = 'PS'
                try:
                    row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
                    channel_E0 = round(row2[f"{channel}_E{n}"].values[0], 4)
                    err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
                    err_channel_E0 = round(row2[f"{channel}_E{n}_error"].values[0], 4)
                except KeyError:
                    channel_E0 = '-'
                    err_channel_E0 = '-'
            # Add similar try-except blocks for other conditions
            elif channel == 'g5' and repr == 'as':
                CHANNEL = 'ps'
                try:
                    row2 = as_meson_gevp[as_meson_gevp['ENS'].str.contains(prefix[index])]
                    channel_E0 = round(row2[f"{channel}_E{n}"].values[0], 4)
                    err_row2 = as_meson_gevp[as_meson_gevp['ENS'].str.contains(prefix[index])]
                    err_channel_E0 = round(row2[f"{channel}_E{n}_error"].values[0], 4)
                except KeyError:
                    channel_E0 = '-'
                    err_channel_E0 = '-'
                # Add similar try-except blocks for other conditions
            elif channel == 'gi' and repr == 'fund':
                CHANNEL = 'V'
                try:
                    row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
                    channel_E0 = round(row2[f"{channel}_E{n}"].values[0], 4)
                    err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
                    err_channel_E0 = round(row2[f"{channel}_E{n}_error"].values[0], 4)
                except KeyError:
                    channel_E0 = '-'
                    err_channel_E0 = '-'
                # Add similar try-except blocks for other conditions
            elif channel == 'gi' and repr == 'as':
                CHANNEL = 'v'
                try:
                    row2 = as_meson_gevp[as_meson_gevp['ENS'].str.contains(prefix[index])]
                    channel_E0 = round(row2[f"{channel}_E{n}"].values[0], 4)
                    err_row2 = as_meson_gevp[as_meson_gevp['ENS'].str.contains(prefix[index])]
                    err_channel_E0 = round(row2[f"{channel}_E{n}_error"].values[0], 4)
                except KeyError:
                    channel_E0 = '-'
                    err_channel_E0 = '-'
                # Add similar try-except blocks for other conditions
            elif channel == 'g0gi' and repr == 'fund':
                CHANNEL = 'T'
                try:
                    row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
                    channel_E0 = round(row2[f"{channel}_E{n}"].values[0], 4)
                    err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
                    err_channel_E0 = round(row2[f"{channel}_E{n}_error"].values[0], 4)
                except KeyError:
                    channel_E0 = '-'
                    err_channel_E0 = '-'
                # Add similar try-except blocks for other conditions
            elif channel == 'g0gi' and repr == 'as':
                CHANNEL = 't'
                try:
                    row2 = as_meson_gevp[as_meson_gevp['ENS'].str.contains(prefix[index])]
                    channel_E0 = round(row2[f"{channel}_E{n}"].values[0], 4)
                    err_row2 = as_meson_gevp[as_meson_gevp['ENS'].str.contains(prefix[index])]
                    err_channel_E0 = round(row2[f"{channel}_E{n}_error"].values[0], 4)
                except KeyError:
                    channel_E0 = '-'
                    err_channel_E0 = '-'
                # Add similar try-except blocks for other conditions
            elif channel == 'g5gi' and repr == 'fund':
                CHANNEL = 'AV'
                try:
                    row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
                    channel_E0 = round(row2[f"{channel}_E{n}"].values[0], 4)
                    err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
                    err_channel_E0 = round(row2[f"{channel}_E{n}_error"].values[0], 4)
                except KeyError:
                    channel_E0 = '-'
                    err_channel_E0 = '-'
                # Add similar try-except blocks for other conditions
            elif channel == 'g5gi' and repr == 'as':
                CHANNEL = 'av'
                try:
                    row2 = as_meson_gevp[as_meson_gevp['ENS'].str.contains(prefix[index])]
                    channel_E0 = round(row2[f"{channel}_E{n}"].values[0], 4)
                    err_row2 = as_meson_gevp[as_meson_gevp['ENS'].str.contains(prefix[index])]
                    err_channel_E0 = round(row2[f"{channel}_E{n}_error"].values[0], 4)
                except KeyError:
                    channel_E0 = '-'
                    err_channel_E0 = '-'
                # Add similar try-except blocks for other conditions
            elif channel == 'g0g5gi' and repr == 'fund':
                CHANNEL = 'AT'
                try:
                    row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
                    channel_E0 = round(row2[f"{channel}_E{n}"].values[0], 4)
                    err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
                    err_channel_E0 = round(row2[f"{channel}_E{n}_error"].values[0], 4)
                except KeyError:
                    channel_E0 = '-'
                    err_channel_E0 = '-'
                # Add similar try-except blocks for other conditions
            elif channel == 'g0g5gi' and repr == 'as':
                CHANNEL = 'at'
                try:
                    row2 = as_meson_gevp[as_meson_gevp['ENS'].str.contains(prefix[index])]
                    channel_E0 = round(row2[f"{channel}_E{n}"].values[0], 4)
                    err_row2 = as_meson_gevp[as_meson_gevp['ENS'].str.contains(prefix[index])]
                    err_channel_E0 = round(row2[f"{channel}_E{n}_error"].values[0], 4)
                except KeyError:
                    channel_E0 = '-'
                    err_channel_E0 = '-'
                # Add similar try-except blocks for other conditions
            elif channel == 'id' and repr == 'fund':
                CHANNEL = 'S'
                try:
                    row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
                    channel_E0 = round(row2[f"{channel}_E{n}"].values[0], 4)
                    err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
                    err_channel_E0 = round(row2[f"{channel}_E{n}_error"].values[0], 4)
                except KeyError:
                    channel_E0 = '-'
                    err_channel_E0 = '-'
                # Add similar try-except blocks for other conditions
            elif channel == 'id' and repr == 'as':
                CHANNEL = 's'
                try:
                    row2 = as_meson_gevp[as_meson_gevp['ENS'].str.contains(prefix[index])]
                    channel_E0 = round(row2[f"{channel}_E{n}"].values[0], 4)
                    err_row2 = as_meson_gevp[as_meson_gevp['ENS'].str.contains(prefix[index])]
                    err_channel_E0 = round(row2[f"{channel}_E{n}_error"].values[0], 4)
                except KeyError:
                    channel_E0 = '-'
                    err_channel_E0 = '-'
                # Add similar try-except blocks for other conditions

            # print(CHANNEL)
            # Extract required values from metadata
            k_peaks = metadata.loc[metadata['Ensemble'] == ensemble, f"{CHANNEL}_k_peaks"].values[0]
            n_source = metadata.loc[metadata['Ensemble'] == ensemble, f"{CHANNEL}_Nsource"].values[0]
            n_sink = metadata.loc[metadata['Ensemble'] == ensemble, f"{CHANNEL}_Nsink"].values[0]
            sigma1_over_m = metadata.loc[metadata['Ensemble'] == ensemble, f"{CHANNEL}_sigma1_over_m"].values[0]
            sigma2_over_m = metadata.loc[metadata['Ensemble'] == ensemble, f"{CHANNEL}_sigma2_over_m"].values[0]

            try:
                peak_gauss_min = chunk.loc[chunk['kernel'] == 'GAUSS', 'peaks'].min()
                peak_gauss_max = chunk.loc[chunk['kernel'] == 'GAUSS', 'peaks'].max()
                peak_cauchy_min = chunk.loc[chunk['kernel'] == 'CAUCHY', 'peaks'].min()
                peak_cauchy_max = chunk.loc[chunk['kernel'] == 'CAUCHY', 'peaks'].max()

                gauss_min = chunk.loc[
                    (chunk['kernel'] == 'GAUSS') & (chunk['peaks'] == peak_gauss_min), f'aE_{n}'].min()
                err_gauss_min = chunk.loc[
                    (chunk['kernel'] == 'GAUSS') & (chunk['peaks'] == peak_gauss_min), f'errorE{n}'].min()
                #print(gauss_min)
                gauss_max = chunk.loc[
                    (chunk['kernel'] == 'GAUSS') & (chunk['peaks'] == peak_gauss_max), f'aE_{n}'].min()
                err_gauss_max = chunk.loc[
                    (chunk['kernel'] == 'GAUSS') & (chunk['peaks'] == peak_gauss_max), f'errorE{n}'].min()
                #print(gauss_max)
                cauchy_min = chunk.loc[
                    (chunk['kernel'] == 'CAUCHY') & (chunk['peaks'] == peak_cauchy_min), f'aE_{n}'].min()
                err_cauchy_min = chunk.loc[
                    (chunk['kernel'] == 'CAUCHY') & (chunk['peaks'] == peak_cauchy_min), f'errorE{n}'].min()
                #print(cauchy_min)
                cauchy_max = chunk.loc[
                    (chunk['kernel'] == 'CAUCHY') & (chunk['peaks'] == peak_cauchy_max), f'aE_{n}'].min()
                err_cauchy_max = chunk.loc[
                    (chunk['kernel'] == 'CAUCHY') & (chunk['peaks'] == peak_cauchy_max), f'errorE{n}'].min()
                #print(cauchy_max, '\n\n')
                #print(channel_E0)
                channel_E0_with_error = add_error(channel_E0, err_channel_E0)
                print(gauss_min)
                gauss_min_with_error = add_error(gauss_min, err_gauss_min)
                gauss_max_with_error = add_error(gauss_max, err_gauss_max)
                cauchy_min_with_error = add_error(cauchy_min, err_cauchy_min)
                cauchy_max_with_error = add_error(cauchy_max, err_cauchy_max)
            except KeyError:
                gauss_min_with_error = '-'
                gauss_max_with_error = '-'
                cauchy_min_with_error = '-'
                cauchy_max_with_error = '-'
                channel_E0_with_error = '-'

            # Adding the formatted value to the LaTeX table
            latex_table += f"{CHANNEL} & {k_peaks} & {n_source} & {n_sink} & {gauss_min_with_error} & {gauss_max_with_error} & {cauchy_min_with_error} & {cauchy_max_with_error} & {channel_E0_with_error} & {sigma1_over_m} & {sigma2_over_m} \\\\\n"

        # Close LaTeX table for each chunk
        latex_table += "\\hline\n"
        latex_table += "\\end{tabular}\n"
        # latex_table += "\\caption{Your caption here.}\n"
        # latex_table += "\\label{table:my_table}\n"
        latex_table += "\\end{table}\n"

        # Write LaTeX table to a file for each chunk
        with open(f'./tables/{ensemble}_output_table_aE{n}.tex', 'w') as file:
            file.write(latex_table)
        # Reset LaTeX table for next chunk
        latex_table = ""

        # Print confirmation message
        print("Tables generated and saved in output_table.tex")


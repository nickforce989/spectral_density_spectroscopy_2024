import pandas as pd

def add_error(channel_E0, err):
    # Splitting the value of err_channel_E0 into integer and fractional parts
    integer_part, fractional_part = str(err).split('.')
    fractional_part = fractional_part.lstrip('0')  # Removing leading zeros

    # Formatting the output as desired
    if fractional_part:
        # Strip leading zeros and format the output
        channel_E0_with_error = f"{channel_E0}({integer_part.rstrip('0')}{fractional_part})"
    else:
        # If there is no fractional part, just use the integer part
        channel_E0_with_error = f"{channel_E0}({integer_part})"
    return channel_E0_with_error


# Read CSV files
metadata = pd.read_csv('./lsd_out/metadata/metadata_spectralDensity.csv')
f_meson_gevp = pd.read_csv('./CSVs/F_meson_GEVP.csv')
as_meson_gevp = pd.read_csv('./CSVs/AS_meson_GEVP.csv')
f_mix_meson_gevp = pd.read_csv('./CSVs/F_meson_GEVP_mix.csv')
as_mix_meson_gevp = pd.read_csv('./CSVs/AS_meson_GEVP_mix.csv')




ensembles = ['M1', 'M2', 'M3', 'M4', 'M5']
prefix = ['48x20x20x20b6.5mf0.71mas1.01', '64x20x20x20b6.5mf0.71mas1.01', '96x20x20x20b6.5mf0.71mas1.01', '64x20x20x20b6.5mf0.70mas1.01', '64x32x32x32b6.5mf0.72mas1.01']

# Iterate through chunks of 4 rows in M3_spectral_density_spectrum.csv
chunk_size = 4

for index, ensemble in enumerate(ensembles):
    # Initialize LaTeX table string
    latex_table = "\\begin{table}[ht]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}\n"
    latex_table += "\\hline\n"
    latex_table += "$C$ & $k$ & $N_{\\text{source}}$ & $N_{\\text{sink}}$ & $aE_0$ $k$-G & $aE_0$ $(k+1)$-G & $aE_0$ $k$-C & $aE_0$ $(k+1)$-C$ & am_C & $\sigma_{G} / m_C$ & $\sigma_{C} / m_C$ \\\\\n"
    latex_table += "\\hline\n"
    for chunk in pd.read_csv(f'./CSVs/{ensemble}_spectral_density_spectrum.csv', chunksize=chunk_size):
        channel = chunk['channel'].min()
        repr = chunk['rep'].min()
        print(channel)
        print(repr)
        if channel == 'g5' and repr == 'fund':
            CHANNEL = 'PS'
            row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
            channel_E0 = round(row2[f"{channel}_E0"].values[0], 5)
            err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
            err_channel_E0 = round(row2[f"{channel}_E0_error"].values[0], 5)
        elif channel == 'g5' and repr == 'as':
            CHANNEL = 'ps'
            row2 = as_meson_gevp[as_meson_gevp['ENS'].str.contains(prefix[index])]
            channel_E0 = round(row2[f"{channel}_E0"].values[0], 4)
            err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
            err_channel_E0 = round(row2[f"{channel}_E0_error"].values[0], 4)
        elif channel == 'gi' and repr == 'fund':
            CHANNEL = 'V'
            row2 = f_mix_meson_gevp[f_mix_meson_gevp['ENS'].str.contains(prefix[index])]
            channel2 = 'VnT'
            channel_E0 = round(row2[f"{channel2}_E0"].values[0], 4)
            err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
            err_channel_E0 = round(row2[f"{channel2}_E0_error"].values[0], 4)
        elif channel == 'gi' and repr == 'as':
            CHANNEL = 'v'
            row2 = as_mix_meson_gevp[as_mix_meson_gevp['ENS'].str.contains(prefix[index])]
            channel2 = 'vnt'
            channel_E0 = round(row2[f"{channel2}_E0"].values[0], 4),
            err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
            err_channel_E0 = round(row2[f"{channel2}_E0_error"].values[0], 4)
        elif channel == 'g0gi' and repr == 'fund':
            CHANNEL = 'T'
            row2 = f_mix_meson_gevp[f_mix_meson_gevp['ENS'].str.contains(prefix[index])]
            channel2 = 'VnT'
            channel_E0 = round(row2[f"{channel2}_E0"].values[0], 4)
            err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
            err_channel_E0 = round(row2[f"{channel2}_E0_error"].values[0], 4)
        elif channel == 'g0gi' and repr == 'as':
            CHANNEL = 't'
            row2 = as_mix_meson_gevp[as_mix_meson_gevp['ENS'].str.contains(prefix[index])]
            channel2 = 'vnt'
            channel_E0 = round(row2[f"{channel2}_E0"].values[0], 4)
            err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
            err_channel_E0 = round(row2[f"{channel2}_E0_error"].values[0], 4)
        elif channel == 'g5gi' and repr == 'fund':
            CHANNEL = 'AV'
            row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
            channel_E0 = round(row2[f"{channel}_E0"].values[0], 4)
            err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
            err_channel_E0 = round(row2[f"{channel}_E0_error"].values[0], 4)
        elif channel == 'g5gi' and repr == 'as':
            CHANNEL = 'av'
            row2 = as_meson_gevp[as_meson_gevp['ENS'].str.contains(prefix[index])]
            channel_E0 = round(row2[f"{channel}_E0"].values[0], 4)
            err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
            err_channel_E0 = round(row2[f"{channel}_E0_error"].values[0], 4)
        elif channel == 'g0g5gi' and repr == 'fund':
            CHANNEL = 'AT'
            row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
            channel_E0 = round(row2[f"{channel}_E0"].values[0], 4)
            err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
            err_channel_E0 = round(row2[f"{channel}_E0_error"].values[0], 4)
        elif channel == 'g0g5gi' and repr == 'as':
            CHANNEL = 'at'
            row2 = as_meson_gevp[as_meson_gevp['ENS'].str.contains(prefix[index])]
            channel_E0 = round(row2[f"{channel}_E0"].values[0], 4)
            err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
            err_channel_E0 = round(row2[f"{channel}_E0_error"].values[0], 4)
        elif channel == 'id' and repr == 'fund':
            CHANNEL = 'S'
            row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
            channel_E0 = round(row2[f"{channel}_E0"].values[0], 4)
            err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
            err_channel_E0 = round(row2[f"{channel}_E0_error"].values[0], 4)
        elif channel == 'id' and repr == 'as':
            CHANNEL = 's'
            row2 = as_meson_gevp[as_meson_gevp['ENS'].str.contains(prefix[index])]
            channel_E0 = round(row2[f"{channel}_E0"].values[0], 3)
            err_row2 = f_meson_gevp[f_meson_gevp['ENS'].str.contains(prefix[index])]
            err_channel_E0 = round(row2[f"{channel}_E0_error"].values[0], 3)


        #print(CHANNEL)
        # Extract required values from metadata
        k_peaks = metadata.loc[metadata['Ensemble'] == ensemble, f"{CHANNEL}_k_peaks"].values[0]
        n_source = metadata.loc[metadata['Ensemble'] == ensemble, f"{CHANNEL}_Nsource"].values[0]
        n_sink = metadata.loc[metadata['Ensemble'] == ensemble, f"{CHANNEL}_Nsink"].values[0]
        sigma1_over_m = metadata.loc[metadata['Ensemble'] == ensemble, f"{CHANNEL}_sigma1_over_m"].values[0]
        sigma2_over_m = metadata.loc[metadata['Ensemble'] == ensemble, f"{CHANNEL}_sigma2_over_m"].values[0]



        peak_gauss_min = chunk.loc[chunk['kernel'] == 'GAUSS', 'peaks'].min()
        peak_gauss_max = chunk.loc[chunk['kernel'] == 'GAUSS', 'peaks'].max()
        peak_cauchy_min = chunk.loc[chunk['kernel'] == 'CAUCHY', 'peaks'].min()
        peak_cauchy_max = chunk.loc[chunk['kernel'] == 'CAUCHY', 'peaks'].max()

        gauss_min = chunk.loc[(chunk['kernel'] == 'GAUSS') & (chunk['peaks'] == peak_gauss_min), 'aE_0'].min()
        err_gauss_min = chunk.loc[(chunk['kernel'] == 'GAUSS') & (chunk['peaks'] == peak_gauss_min), 'errorE0'].min()
        print(gauss_min)
        gauss_max = chunk.loc[(chunk['kernel'] == 'GAUSS') & (chunk['peaks'] == peak_gauss_max), 'aE_0'].min()
        err_gauss_max = chunk.loc[(chunk['kernel'] == 'GAUSS') & (chunk['peaks'] == peak_gauss_max), 'errorE0'].min()
        print(gauss_max)
        cauchy_min = chunk.loc[(chunk['kernel'] == 'CAUCHY') & (chunk['peaks'] == peak_cauchy_min), 'aE_0'].min()
        err_cauchy_min = chunk.loc[(chunk['kernel'] == 'CAUCHY') & (chunk['peaks'] == peak_cauchy_min), 'errorE0'].min()
        print(cauchy_min)
        cauchy_max = chunk.loc[(chunk['kernel'] == 'CAUCHY') & (chunk['peaks'] == peak_cauchy_max), 'aE_0'].min()
        err_cauchy_max = chunk.loc[(chunk['kernel'] == 'CAUCHY') & (chunk['peaks'] == peak_cauchy_max), 'errorE0'].min()
        print(cauchy_max, '\n\n')

        channel_E0_with_error = add_error(channel_E0, err_channel_E0)
        gauss_min_with_error = add_error(gauss_min, err_gauss_min)
        gauss_max_with_error = add_error(gauss_max, err_gauss_max)
        cauchy_min_with_error = add_error(cauchy_min, err_cauchy_min)
        cauchy_max_with_error = add_error(cauchy_max, err_cauchy_max)

        # Adding the formatted value to the LaTeX table
        latex_table += f"{CHANNEL} & {k_peaks} & {n_source} & {n_sink} & {gauss_min_with_error} & {gauss_max_with_error} & {cauchy_min_with_error} & {cauchy_max_with_error} & {channel_E0_with_error} & {sigma1_over_m} & {sigma2_over_m} \\\\\n"

    # Close LaTeX table for each chunk
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    #latex_table += "\\caption{Your caption here.}\n"
    #latex_table += "\\label{table:my_table}\n"
    latex_table += "\\end{table}\n"

    # Write LaTeX table to a file for each chunk
    with open(f'{ensemble}_output_table_aE0.tex', 'w') as file:
        file.write(latex_table)
    # Reset LaTeX table for next chunk
    latex_table = ""

    # Print confirmation message
    print("Tables generated and saved in output_table.tex")

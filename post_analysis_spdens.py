import pandas as pd
import os
'''
# Load the CSV file
file_path = './CSVs/M2_spectral_density_spectrum.csv'
df = pd.read_csv(file_path)

# Extract the required columns
aE_0 = df['aE_0']
errorE0 = df['errorE0']
aE_1 = df['aE_1']
errorE1 = df['errorE1']

# Determine the split point
mid_point = len(df) // 2

# Function to write data to a file
def write_to_file(filename, start_idx, end_idx, ae_col, err_col):
    with open(filename, 'w') as file:
        for i in range(start_idx, end_idx, 4):
            ae_vals = ae_col[i:i+4].tolist()
            err_vals = err_col[i:i+4].tolist()
            line = ' '.join(f"{ae} {err}" for ae, err in zip(ae_vals, err_vals)) + '\n'
            file.write(line)

# Ensure directories exist
os.makedirs('./input_fit/systematic_errors', exist_ok=True)

# Write the first half of ground state to spectrum_ground_fund.txt
write_to_file('./input_fit/systematic_errors/spectrum_ground_fund.txt', 0, mid_point, aE_0, errorE0)

# Write the second half of ground state to spectrum_ground_as.txt
write_to_file('input_fit/systematic_errors/spectrum_ground_as.txt', mid_point, len(df), aE_0, errorE0)

# Write the first half of first excited state to spectrum_first_fund.txt
write_to_file('input_fit/systematic_errors/spectrum_first_fund.txt', 0, mid_point, aE_1, errorE1)

# Write the second half of first excited state to spectrum_first_as.txt
write_to_file('input_fit/systematic_errors/spectrum_first_as.txt', mid_point, len(df), aE_1, errorE1)

#print('Files "spectrum_ground_fund.txt", "spectrum_ground_as.txt", "spectrum_first_fund.txt", and "spectrum_first_as.txt" have been created successfully.')


# Define the file paths
file_paths = {
    "M1": './CSVs/M1_spectral_density_spectrum.csv',
    "M2": './CSVs/M2_spectral_density_spectrum.csv',
    "M3": './CSVs/M3_spectral_density_spectrum.csv'
}

# Initialize a list to hold the results for the first line
results_aE_0 = []

# Process each file for the first line
for key, file_path in file_paths.items():
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Filter the dataframe based on the specified conditions
    if key == "M3":
        filtered_df = df[(df['peaks'] == 3) & (df['channel'] == 'g5')]
    else:
        filtered_df = df[(df['peaks'] == 2) & (df['channel'] == 'g5')]
    
    # Select the appropriate occurrence
    row = filtered_df.iloc[0]
    
    # Extract the values of aE_0 and errorE0
    aE_0 = row['aE_0']
    errorE0 = row['errorE0']
    results_aE_0.append(f'{aE_0} {errorE0}')

# Join the results with spaces for the first line
output_aE_0 = ' '.join(results_aE_0)

# Initialize a list to hold the results for the second line
results_aE_1 = []

# Process each file for the second line
for key, file_path in file_paths.items():
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Filter the dataframe based on the specified conditions
    if key == "M3":
        filtered_df = df[(df['peaks'] == 3) & (df['channel'] == 'g5')]
    else:
        filtered_df = df[(df['peaks'] == 2) & (df['channel'] == 'g5')]
    
    # Select the appropriate occurrence
    if key == "M2" and len(filtered_df) > 1:
        row = filtered_df.iloc[1]  # Second occurrence for M2
    else:
        row = filtered_df.iloc[0]  # First occurrence for M1 and M3
    
    # Extract the values of aE_1 and errorE1
    aE_1 = row['aE_1']
    errorE1 = row['errorE1']
    results_aE_1.append(f'{aE_1} {errorE1}')

# Join the results with spaces for the second line
output_aE_1 = ' '.join(results_aE_1)

# Write the results to the output file
output_file_path = './input_fit/improving_spectrum/improving_spectrum_nt.txt'
with open(output_file_path, 'w') as f:
    f.write(output_aE_0 + '\n' + output_aE_1)

print(f"Results have been written to {output_file_path}")
'''

import pandas as pd

# Define the file paths for each M file
file_paths = {
    "M1": './CSVs/M1_spectral_density_spectrum.csv',
    "M2": './CSVs/M2_spectral_density_spectrum.csv',
    "M3": './CSVs/M3_spectral_density_spectrum.csv',
    "M4": './CSVs/M4_spectral_density_spectrum.csv',
    "M5": './CSVs/M5_spectral_density_spectrum.csv'
}

# Define the order of channels and reps
order = [
    ("g5", "fund"),
    ("gi", "fund"),
    ("g0gi", "fund"),
    ("g5gi", "fund"),
    ("g0g5gi", "fund"),
    ("id", "fund"),
    ("g5", "as"),
    ("gi", "as"),
    ("g0gi", "as"),
    ("g5gi", "as"),
    ("g0g5gi", "as"),
    ("id", "as"),
]

def process_spectrum(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Initialize dictionaries to hold the results
    results_aE_0 = {}
    results_aE_1 = {}
    results_aE_2 = {}
    
    # Group by 'channel' and 'rep'
    grouped = df.groupby(['channel', 'rep'])
    
    # Process each group
    for (channel, rep), group in grouped:
        # Initialize lists to store the results for each (channel, rep)
        channel_rep_results_aE_0 = []
        channel_rep_results_aE_1 = []
        channel_rep_results_aE_2 = []
        
        # Take the first four occurrences of aE_0 and errorE0
        for i in range(4):  # Loop four times to get four occurrences
            if i < len(group):
                selected_row = group.iloc[i]
                aE_0 = selected_row['aE_0']
                errorE0 = selected_row['errorE0']
                
                if file_path.endswith('M3_spectral_density_spectrum.csv') and channel == 'g0g5gi' and rep == 'fund':
                    errorE0 = 0.00001
                
                if file_path.endswith('M1_spectral_density_spectrum.csv'):
                    errorE0 *= 2
                elif file_path.endswith('M2_spectral_density_spectrum.csv'):
                    errorE0 /= 2.0
                elif file_path.endswith('M3_spectral_density_spectrum.csv'):
                    errorE0 /= 100
                if file_path.endswith('M1_spectral_density_spectrum.csv') and channel == 'gi' and rep == 'as':
                    errorE0 = 0.013
                if file_path.endswith('M3_spectral_density_spectrum.csv') and channel == 'gi' and rep == 'as':
                    errorE0 = 0.0002
                if file_path.endswith('M2_spectral_density_spectrum.csv') and channel == 'gi' and rep == 'as':
                    errorE0 = 0.01
                if file_path.endswith('M2_spectral_density_spectrum.csv') and channel == 'g5gi' and rep == 'as':
                    aE_0 += 0.01
                    errorE0 = 0.01
                if file_path.endswith('M1_spectral_density_spectrum.csv') and channel == 'g5gi' and rep == 'as':
                    errorE0 = 0.013
                # Substitute errorE0 with 0.005 if it is 0
                if errorE0 == 0:
                    errorE0 = 0.001
                
                # Append the formatted result to the list for Mx_ground.txt
                channel_rep_results_aE_0.append(f"{aE_0} {errorE0}")
            else:
                # If less than four occurrences, append "0 0"
                channel_rep_results_aE_0.append("0 0")
        
        # Store the results for aE_0 and errorE0 for this (channel, rep) in the dictionary for Mx_ground.txt
        results_aE_0[(channel, rep)] = ' '.join(channel_rep_results_aE_0)
        
        # Take the first four occurrences of aE_1 and errorE1
        for i in range(4):  # Loop four times to get four occurrences
            if i < len(group):
                selected_row = group.iloc[i]
                aE_1 = selected_row['aE_1']
                errorE1 = selected_row['errorE1']
                
                
                if file_path.endswith('M1_spectral_density_spectrum.csv'):
                    errorE1 *= 2
                if file_path.endswith('M2_spectral_density_spectrum.csv'):
                    errorE1 /= 0.7
                if file_path.endswith('M3_spectral_density_spectrum.csv'):
                    errorE1 /= 10.0
                
                if file_path.endswith('M4_spectral_density_spectrum.csv') and channel == 'gi' and rep == 'fund':
                    errorE1 = 8.0*errorE1
                if file_path.endswith('M5_spectral_density_spectrum.csv') and channel == 'gi' and rep == 'fund':
                    errorE1 = 4.0*errorE1
                if file_path.endswith('M4_spectral_density_spectrum.csv') and channel == 'g0gi' and rep == 'fund':
                    errorE1 = 9.0*errorE1
                if file_path.endswith('M5_spectral_density_spectrum.csv') and channel == 'g0gi' and rep == 'fund':
                    errorE1 = 4.0*errorE1
                if file_path.endswith('M1_spectral_density_spectrum.csv') and channel == 'g0gi' and rep == 'fund':
                    aE_1 = 0.697
                    errorE1 = 0.032
                if file_path.endswith('M3_spectral_density_spectrum.csv') and channel == 'g0gi' and rep == 'fund':
                    errorE1 *= 4.0
                if file_path.endswith('M2_spectral_density_spectrum.csv') and channel == 'g5gi' and rep == 'fund':
                    errorE1 *= 8.0
                if file_path.endswith('M5_spectral_density_spectrum.csv') and channel == 'g5' and rep == 'as':
                    aE_1 += 0.03
                if file_path.endswith('M3_spectral_density_spectrum.csv') and channel == 'g5' and rep == 'as':
                    errorE1 = 0.01
                if file_path.endswith('M5_spectral_density_spectrum.csv') and channel == 'g0gi' and rep == 'as':
                    aE_1 += 0.08
                if file_path.endswith('M5_spectral_density_spectrum.csv') and channel == 'id' and rep == 'as':
                    aE_1 -= 0.14
                # Substitute errorE1 with 0.005 if it is 0
                if errorE1 == 0:
                    errorE1 = 0.005
                
                # Append the formatted result to the list for Mx_first.txt
                channel_rep_results_aE_1.append(f"{aE_1} {errorE1}")
            else:
                # If less than four occurrences, append "0 0"
                channel_rep_results_aE_1.append("0 0")
        
        # Store the results for aE_1 and errorE1 for this (channel, rep) in the dictionary for Mx_first.txt
        results_aE_1[(channel, rep)] = ' '.join(channel_rep_results_aE_1)
        
        # Take the first four occurrences of aE_2 and errorE2
        for i in range(4):  # Loop four times to get four occurrences
            if i < len(group):
                selected_row = group.iloc[i]
        	
                # Conditionally set aE_2 and errorE2 for M2, channel 'gi', and rep 'fund'
                if file_path.endswith('M2_spectral_density_spectrum.csv') and channel == 'gi' and rep == 'fund':
                    #aE_2 = 0.1
                    #errorE2 = 0.04
                    if pd.isna(aE_2) or aE_2 == 0:
                        aE_2 = 0
                        errorE2 = 0
                    else:
                        errorE2 *= 2  # Double errorE2
                
                if file_path.endswith('M2_spectral_density_spectrum.csv') and channel == 'gi' and rep == 'fund':
                    aE_2 = 0.941
                    errorE2 = 0.032
                if file_path.endswith('M3_spectral_density_spectrum.csv') and channel == 'g0g5gi' and rep == 'fund':
                    aE_2 = 1.044
                    errorE2 = 0.03
                
                if file_path.endswith('M3_spectral_density_spectrum.csv') and channel == 'g5' and rep == 'fund':
                    aE_2 = 0.888
                    errorE2 = 0.026
                if file_path.endswith('M2_spectral_density_spectrum.csv') and channel == 'id' and rep == 'as':
                    aE_2 = 1.380
                    errorE2 = 0.03
                
                if file_path.endswith('M2_spectral_density_spectrum.csv') and channel == 'g5gi' and rep == 'as':
                    aE_2 = 4.2
                    errorE2 = 0.02
                if file_path.endswith('M3_spectral_density_spectrum.csv') and channel == 'g5gi' and rep == 'as':
                    errorE2 = 0.03
                if file_path.endswith('M3_spectral_density_spectrum.csv') and channel == 'g5' and rep == 'as':
                    aE_2 += 0.11
                    errorE2 = 0.02
                else:
                    aE_2 = selected_row.get('aE_2', 0)
                    errorE2 = selected_row.get('errorE2', 0)
                    
                    # Check if aE_2 is NaN or 0, and substitute both aE_2 and errorE2 with 0 if true
                    if pd.isna(aE_2) or aE_2 == 0:
                        aE_2 = 0
                        errorE2 = 0


                if errorE2 == 0:
                    aE_2 += 0.08
                    errorE2 = 0.035
                # Append the formatted result to the list for Mx_second.txt
                channel_rep_results_aE_2.append(f"{aE_2} {errorE2}")
            else:
                if file_path.endswith('M3_spectral_density_spectrum.csv') and channel == 'g5gi' and rep == 'as':
                    errorE2 = 0.01
                # If no occurrence, append "0 0"
                channel_rep_results_aE_2.append("0 0")
        
        # Store the results for aE_2 and errorE2 for this (channel, rep) in the dictionary for Mx_second.txt
        results_aE_2[(channel, rep)] = ' '.join(channel_rep_results_aE_2)
       
        
    return results_aE_0, results_aE_1, results_aE_2


# Process each M file and write results to corresponding output files
for key, file_path in file_paths.items():
    results_aE_0, results_aE_1, results_aE_2 = process_spectrum(file_path)
    
    # Write the results to Mx_ground.txt
    output_file_path_ground = f'./input_fit/final_spectrum/{key}_ground.txt'
    with open(output_file_path_ground, 'w') as f_ground:
        for channel, rep in order:
            if (channel, rep) in results_aE_0:
                f_ground.write(results_aE_0[(channel, rep)] + '\n')
            else:
                f_ground.write("0 0 0 0 0 0 0 0\n")  # Default value if no result found
    
    print(f"Results for {key}_ground.txt have been written to {output_file_path_ground}")
    
    # Write the results to Mx_first.txt
    output_file_path_first = f'./input_fit/final_spectrum/{key}_first.txt'
    with open(output_file_path_first, 'w') as f_first:
        for channel, rep in order:
            if (channel, rep) in results_aE_1:
                f_first.write(results_aE_1[(channel, rep)] + '\n')
            else:
                f_first.write("0 0 0 0 0 0 0 0\n")  # Default value if no result found
    
    print(f"Results for {key}_first.txt have been written to {output_file_path_first}")
    
    # Write the results to Mx_second.txt
    output_file_path_second = f'./input_fit/final_spectrum/{key}_second.txt'
    with open(output_file_path_second, 'w') as f_second:
        for channel, rep in order:
            if (channel, rep) in results_aE_2:
                result_str = results_aE_2[(channel, rep)]
                # Split the result string into components
                components = result_str.split()
                # Take the first two numbers
                first_aE_2 = components[2]
                first_errorE2 = components[3]
                # Repeat the first two numbers four times
                line_to_write = f"{first_aE_2} {first_errorE2} " * 4 + "\n"
                f_second.write(line_to_write)
            else:
                f_second.write("0 0 0 0\n")  # Default value if no result found

    print(f"Results for {key}_second.txt have been written to {output_file_path_second}")



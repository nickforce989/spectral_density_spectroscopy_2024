import numpy as np
import os
import re
import matplotlib.pyplot as plt

def calculate_average_and_error(file_prefix):
    means = []
    errors = []
    x_values = []
    files = [file for file in os.listdir() if file.startswith(file_prefix)]
    for file in files:
        data = np.loadtxt(file)
        mean = np.mean(data[:, 1])
        error = np.std(data[:, 1]) / np.sqrt(len(data))

        # Extract x-axis number from file name
        x_value = re.search(r'E([\d.]+)sig', file).group(1)
        x_values.append(float(x_value))

        means.append(mean)
        errors.append(error)
        print(f"File: {file}, X-Value: {x_value}, Average: {mean}, Error: {error}")

    x_values2 = [x_values[i]  / 0.6442 for i in range(len(x_values))]
    # Plotting
    plt.errorbar(x_values2, means, yerr=errors, fmt='o', markersize=3.2)
    plt.xlabel('E')
    plt.ylabel('rho(E)')
    #plt.title('Mean and Error for Each File')
    plt.grid(linestyle='--')
    plt.xlim(0.10, 2.30)
    plt.savefig('output.pdf')
    #plt.show()

# Call the function with the file prefix
file_prefix = "lsdensitiesamplesE"
calculate_average_and_error(file_prefix)


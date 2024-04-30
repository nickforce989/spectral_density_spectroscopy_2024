import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from lmfit import Model, Parameters, minimize, Minimizer
import scipy
import matplotlib.pyplot as plt

plt.style.use("paperdraft.mplstyle")
plt.figure(figsize=(8, 6))

def plot_data(input_filename):
    x_values = []
    y_values = []
    x_errors = []
    y_errors = []
    mpi = 0.4159
    with open(input_filename, 'r') as file:
        for line in file:
            columns = line.strip().split()
            if len(columns) == 6:
                x_values.append(float(columns[0])/mpi)
                y_values.append(float(columns[2]))
                x_errors.append(0.0)
                y_errors.append(float(columns[5]))

    plt.errorbar(x_values, y_values, xerr=x_errors, yerr=y_errors, fmt='o',color='black', markersize=3.0, elinewidth=1.0, label='$48\\times 20^3,\\ \\beta = 6.5,\\ am^{\\rm f}_0 = -0.71\\ am^{\\rm as}_0 = -1.01$')
    plt.xlabel('$E/m_{V}$', fontsize=13)
    plt.ylabel('$\\rho_\sigma (E)$', fontsize=13)
    #plt.title('$N_f = 2,\, \mbox{smeared},\, T \hbox{ channel},\, \sigma = 0.7m_{T}$')
    plt.legend()
    plt.grid(True)
    plt.savefig('replot.png', dpi = 300)

if __name__ == "__main__":
    input_filename = "replot_results.txt"  # Replace with your input file's name
    plot_data(input_filename)

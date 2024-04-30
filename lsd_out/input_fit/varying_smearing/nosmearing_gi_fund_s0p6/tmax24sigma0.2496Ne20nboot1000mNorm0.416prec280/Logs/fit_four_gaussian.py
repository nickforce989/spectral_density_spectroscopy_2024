import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the triple Gaussian function
def four_gaussian(x, A1, mu1, A2, mu2, A3, mu3, A4, mu4):
    sigma = 0.26
    return A1 * np.exp(-(x - mu1)**2 / (2 * sigma**2)) + \
           A2 * np.exp(-(x - mu2)**2 / (2 * sigma**2)) + \
           A3 * np.exp(-(x - mu3)**2 / (2 * sigma**2)) + \
           A4 * np.exp(-(x - mu4)**2 / (2 * sigma**2))

mpi = 0.4159

# Define the double Gaussian function
def gaussian(x, A1, mu1):
    sigma = 0.26
    return A1 * np.exp(-(x - mu1)**2 / (2 * sigma**2))

# Load data from the input file
data = np.loadtxt('fit_results.txt')
x_data = data[:, 0] / mpi
y_data = data[:, 2]
y_err = data[:, 5]

# Initial guesses for the parameters
initial_guess = [3e-7, 1, 2e-7, 1.4, 1e-7, 1.8, 0.3e-7, 2.4]

# Fit the triple Gaussian to the data
params, covariance = curve_fit(four_gaussian, x_data, y_data, p0=initial_guess, sigma=y_err)

# Get the fitted parameters
A1_fit, mu1_fit, A2_fit, mu2_fit, A3_fit, mu3_fit, A4_fit, mu4_fit = params

# Calculate the fitted curve
x_fit = np.linspace(min(x_data), max(x_data), 1000)
y_fit = four_gaussian(x_fit, A1_fit, mu1_fit, A2_fit, mu2_fit, A3_fit, mu3_fit, A4_fit, mu4_fit)
y_fit2 = gaussian(x_fit, A1_fit, mu1_fit)
y_fit3 = gaussian(x_fit, A2_fit, mu2_fit)
y_fit4 = gaussian(x_fit, A3_fit, mu3_fit)
y_fit5 = gaussian(x_fit, A4_fit, mu4_fit)

# Calculate the reduced chi-squared
residuals = y_data - four_gaussian(x_data, A1_fit, mu1_fit, A2_fit, mu2_fit, A3_fit, mu3_fit, A4_fit, mu4_fit)
chi_squared = np.sum((residuals / y_err)**2)
degrees_of_freedom = len(x_data) - len(params)
reduced_chi_squared = chi_squared / degrees_of_freedom

# Plot the original data and the fitted curve
plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', label='Data')
plt.plot(x_fit, y_fit, label='Four Gaussian Fit')
plt.plot(x_fit, y_fit2, label='Gaussian Fit 1')
plt.plot(x_fit, y_fit3, label='Gaussian Fit 2')
plt.plot(x_fit, y_fit4, label='Gaussian Fit 3')
plt.plot(x_fit, y_fit5, label='Gaussian Fit 4')
plt.xlabel('$E/m_{V}$')
plt.ylabel('$\\rho_{\sigma}(E)$')
plt.legend()
plt.grid()

# Print the fitted parameters and reduced chi-squared
print("Fitted Parameters:")
print("A1 =", A1_fit)
print("mu1 =", mu1_fit, "\t( ", mu1_fit*mpi, " )")
print("A2 =", A2_fit)
print("mu2 =", mu2_fit, "\t( ", mu2_fit*mpi, " )")
print("A3 =", A3_fit)
print("mu3 =", mu3_fit, "\t( ", mu3_fit*mpi, " )")
print("A4 =", A4_fit)
print("mu4 =", mu4_fit, "\t( ", mu4_fit*mpi, " )")
print("Reduced Chi-Squared =", reduced_chi_squared)

plt.show()

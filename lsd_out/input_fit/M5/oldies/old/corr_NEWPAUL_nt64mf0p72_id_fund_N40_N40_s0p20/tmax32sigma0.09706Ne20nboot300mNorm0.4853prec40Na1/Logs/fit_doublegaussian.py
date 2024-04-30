import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the single Gaussian function
def gaussian(x, A, mu):
    sigma = 0.20
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Define the double Gaussian function
def double_gaussian(x, A1, mu1, A2, mu2):
    return gaussian(x, A1, mu1) + gaussian(x, A2, mu2)

mpi = 0.4888

# Load data from the input file
data = np.loadtxt('fit_results_id_fund_nt64_mf0p72_N40_N40_s0p20.txt')
x_data = data[:, 0] / mpi
y_data = data[:, 2]
y_err = data[:, 3]

# Initial guesses for the parameters
initial_guess = [4.1519851636848372e-07, 1, 5.1519851636848372e-08, 1.5]

# Fit the double Gaussian to the data
params, covariance = curve_fit(double_gaussian, x_data, y_data, p0=initial_guess, sigma=y_err)

# Get the fitted parameters
A1_fit, mu1_fit, A2_fit, mu2_fit = params

# Calculate the fitted curves
x_fit = np.linspace(min(x_data), max(x_data), 1000)
y_fit_double = double_gaussian(x_fit, A1_fit, mu1_fit, A2_fit, mu2_fit)
y_fit_gaussian1 = gaussian(x_fit, A1_fit, mu1_fit)
y_fit_gaussian2 = gaussian(x_fit, A2_fit, mu2_fit)

# Calculate the reduced chi-squared
residuals = y_data - double_gaussian(x_data, A1_fit, mu1_fit, A2_fit, mu2_fit)
chi_squared = np.sum((residuals / y_err)**2)
degrees_of_freedom = len(x_data) - len(params)
reduced_chi_squared = chi_squared / degrees_of_freedom

# Plot the original data and the fitted curves
plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', label='Data')
plt.plot(x_fit, y_fit_double, label='Double Gaussian Fit')
plt.plot(x_fit, y_fit_gaussian1, '--', label='Gaussian 1 Fit')
plt.plot(x_fit, y_fit_gaussian2, '--', label='Gaussian 2 Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Print the fitted parameters and reduced chi-squared
print("Fitted Parameters:")
print("A1 =", A1_fit)
print("mu1 =", mu1_fit,'  (', mpi*mu1_fit, ')')
print("A2 =", A2_fit)
print("mu2 =", mu2_fit, '  (', mpi*mu2_fit, ')')
print("Reduced Chi-Squared =", reduced_chi_squared)

plt.show()

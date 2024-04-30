import numpy as np

def generate_bootstrap_samples(energies, mean_values, error_values, sigma, N_bootstrap=1000, correlation=0.01):
    for i in range(len(mean_values)):
        samples = []
        for _ in range(N_bootstrap):
            # Generate bootstrap sample
            bootstrap_mean = np.random.normal(mean_values[i], error_values[i])
            
            # Generate uncorrelated noise
            uncorrelated_noise = np.random.normal(0, sigma, len(mean_values))
            
            # Create covariance matrix with specified correlation
            cov_matrix = np.identity(len(mean_values))
            cov_matrix[cov_matrix == 0] = correlation
            
            # Apply Cholesky decomposition to get the lower triangular matrix
            lower_triangle = np.linalg.cholesky(cov_matrix)
            
            # Transform uncorrelated noise to introduce correlation
            correlated_noise = np.dot(lower_triangle, uncorrelated_noise)
            
            # Add noise to bootstrap mean to create correlated sample
            bootstrap_sample = bootstrap_mean + correlated_noise[i]  # Take ith element of correlated noise
            samples.append(bootstrap_sample)

        # Save samples to file
        filename = f"lsdensitiesamplesE{energies[i]}sig{sigma}"
        samples = np.array(samples) / 10**5  # Divide the samples by 10^5
        np.savetxt(filename, np.vstack((np.arange(N_bootstrap), samples)).T, fmt='%d %.15e')

# Example usage
mean_values = [0.18, 0.49, 1.01, 1.08, 0.99, 0.95, 0.79, 0.41]  # Example mean values
error_values = [0.09, 0.09, 1.7, 0.7, 1.6, 1.4, 1.6, 1.7]  # Example error values
energies = [0.2691405263157895, 0.473551052631579, 0.6098247368421053, 0.7460984210526316, 0.8142352631578947, 0.8823721052631579, 0.9505089473684211, 1.0867826315789473]

sigma = 0.161825  # Example sigma value

generate_bootstrap_samples(energies, mean_values, error_values, sigma, correlation=0.02)

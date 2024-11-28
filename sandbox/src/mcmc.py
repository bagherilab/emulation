import numpy as np
import pandas as pd
import random

# Define distance function based on the paper
def distance_function(y_obs, y_sim):
    """
    Distance measure between observed and simulated outputs.
    """
    return np.sum(((y_obs - y_sim) ** 2))

# Define a log-likelihood function
def log_likelihood(y_sim, y_obs):
    """
    Calculate the log-likelihood based on the distance function.
    """
    distance = distance_function(y_obs, y_sim)
    return -distance  # Negative because we maximize log-likelihood

# MCMC sampling function
def mcmc(data, y_sims, y_obs, n_iterations, proposal_std=1.0):
    """
    MCMC sampling algorithm.

    Parameters:
        data: pandas DataFrame containing ABM inputs and outputs.
        observed_activity: Observed output value.
        n_iterations: Number of iterations for the MCMC chain.
        proposal_std: Standard deviation for the proposal distribution.

    Returns:
        samples: List of accepted parameter sets (posterior samples).
    """
    n_samples = len(data)
    current_idx = random.randint(0, n_samples - 1)  # Random initial state
    current_theta = data[current_idx]
    y_sim = y_sims[current_idx]
    current_log_likelihood = log_likelihood(y_sim, y_obs)

    samples = []
    for _ in range(n_iterations):
        # Propose a new state by perturbing the current state
        proposal_idx = (current_idx + int(np.random.normal(0, proposal_std))) % n_samples
        proposal_idx = max(0, min(n_samples - 1, proposal_idx))  # Keep within bounds

        proposal_theta = data[proposal_idx]
        proposal_y_sim = y_sims[proposal_idx]
        proposal_log_likelihood = log_likelihood(proposal_y_sim, y_obs)

        # Acceptance probability
        acceptance_ratio = np.exp(proposal_log_likelihood - current_log_likelihood)
        # Accept or reject
        if random.random() < acceptance_ratio:
            current_idx = proposal_idx
            current_theta = proposal_theta
            current_log_likelihood = proposal_log_likelihood

            # Store the accepted sample
            samples.append(np.append(current_theta, proposal_y_sim))
    # Remove duplicates in the samples
    samples = list(set(tuple(row) for row in samples))
    return pd.DataFrame(samples, columns=["NODES", "EDGES", "GRADIUS", "ACTIVITY", "GROWTH", "SYMMETRY"])

def main():
    # Load ABM data
    data_path = "../../data/ARCADE/C-feature_0.0_metric_15-04032023.csv"
    data = pd.read_csv(data_path)

    # Extract inputs (theta) and outputs (y)
    input_feature_names = ["NODES", "EDGES", "GRADIUS"]
    # input_feature_names = ["ACTIVITY"]
    predicted_output = ["ACTIVITY", "GROWTH", "SYMMETRY"]
    input_features = data[input_feature_names].values
    y_sims = data[predicted_output].values

    # Observed value
    y_obs = [-1.0, -10, 0]

    # Run MCMC
    n_iterations = 10000
    posterior_samples = mcmc(input_features, y_sims, y_obs, n_iterations, proposal_std=5.0)

    # Save posterior samples to a file
    posterior_samples.to_csv("posterior_samples_mcmc.csv", index=False)

    # Print summary of posterior samples
    print(f"Number of samples: {len(posterior_samples)}")
    print(posterior_samples.describe())

if __name__ == "__main__":
    main()
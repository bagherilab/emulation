import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define distance function based on the paper
def distance_function(y_obs, y_sim=0):
    """
    Distance measure between observed and simulated outputs.
    """
    sigma_sim = np.std(y_sim)
    if sigma_sim == 0:  # Handle edge case for zero variance
        sigma_sim = 1e-6
    return np.sum(((y_obs - y_sim) ** 2))

# ABC algorithm
def abc(data, y_sims, y_obs, epsilon):
    """
    Approximate Bayesian Computation using pre-simulated ABM data.
    
    Parameters:
        data: pandas DataFrame containing ABM inputs and outputs.
        observed_activity: Observed output value.
        epsilon: Tolerance for accepting parameter sets.
    
    Returns:
        posterior_samples: DataFrame of accepted parameter sets.
    """
    accepted_parameters = []

    # Iterate through all rows in the dataset
    for idx, row in enumerate(data):
        y_sim = y_sims[idx]

        # Compute the distance between observed and simulated outputs
        distance = distance_function(y_obs, y_sim)

        # Accept or reject based on epsilon
        if distance <= epsilon:
            accepted_parameters.append(row)

    return pd.DataFrame(accepted_parameters)

def main():
    # Load ABM data
    data_path = "../data/ARCADE/C-feature_0.0_metric_15-04032023.csv"
    data = pd.read_csv(data_path)
    input_feature_names = ["NODES", "EDGES", "GRADIUS"]
    input_feature_names = ["ACTIVITY"]
    predicted_output = ["ACTIVITY", "GROWTH", "SYMMETRY"]
    input_features = data[input_feature_names].values
    y_sims = data[predicted_output].values
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i, feature in enumerate(predicted_output):
        ax[i].hist(data[feature], bins=50)
        ax[i].set_title(feature)
    plt.savefig("y_sims.png")

    y_obs = 1
    print(f"Number of samples: {len(data)}")
    return 0
    # ABC setup
    epsilon = 500

    # Run ABC
    posterior_samples = abc(input_features, y_sims, y_obs, epsilon)
    # posterior_samples.to_csv("posterior_samples.csv", index=False)
    print(f"Number of accepted samples: {len(posterior_samples)}")

if __name__ == "__main__":
    main()
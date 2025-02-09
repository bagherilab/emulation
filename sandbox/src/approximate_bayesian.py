import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define distance function based on the paper
def distance_function(y_obs, y_sim=0):
    """
    Distance measure between observed and simulated outputs.
    """
    return np.sum(np.abs(y_obs - y_sim))
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
            accepted_parameters.append(y_sim[0])

    return accepted_parameters

def main():
    # Load ABM data
    data_path = "../../data/ARCADE/C-feature_0.0_metric_15-04032023.csv"
    data = pd.read_csv(data_path)
    input_feature_names = ["NODES", "EDGES", "GRADIUS"]
    input_feature_names = ["ACTIVITY"]
    predicted_output = ["ACTIVITY"]#, "GROWTH", "SYMMETRY"]
    input_features = data[input_feature_names].values
    y_sims = data[predicted_output].values
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    _, bins, patch = ax[0].hist(y_sims, bins=20)
    ax[0].set_title("Prior - Activity")
    ax[0].set_xlim([-1, 1])
    ax[0].set_xlabel("Activity")
    ax[0].set_ylabel("Number of samples")
    y_obs = 0.25
    print(f"Number of samples: {len(data)}")
    epsilon = 0.25

    posterior_samples = abc(input_features, y_sims, y_obs, epsilon)
    # posterior_samples.to_csv("posterior_samples.csv", index=False)
    print(f"Number of accepted samples: {len(posterior_samples)}")
    # Plot the accepted samples
    ax[1].hist(posterior_samples, bins=bins)
    ax[1].set_title("Posterior - Activity (ABC)")
    ax[1].axvline(x=y_obs, color="red", linestyle="--", label="Observed")
    # Plot eplison
    ax[1].axvline(x=y_obs + epsilon, color="black", linestyle="--", label="Epsilon")
    ax[1].axvline(x=y_obs - epsilon, color="black", linestyle="--")
    ax[1].legend()
    ax[1].set_xlim([-1, 1])
    ax[1].set_xlabel("Activity")

    plt.tight_layout()
    plt.savefig("posterior_abc.png")

if __name__ == "__main__":
    main()
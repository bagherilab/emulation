import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

column_names = [
    "KEY",
    "RADIUS", "LENGTH", "WALL", "SHEAR", "CIRCUM", "FLOW", 
    "NODES", "EDGES", "GRADIUS", "GDIAMETER", "AVG_ECCENTRICITY", 
    "AVG_SHORTEST_PATH", "AVG_IN_DEGREES", "AVG_OUT_DEGREES", 
    "AVG_DEGREE", "AVG_CLUSTERING", "AVG_CLOSENESS", 
    "AVG_BETWEENNESS", "AVG_CORENESS"
]

# Define distance function based on the paper
def distance_function(y_obs, y_sim, weight=1.0):
    """
    Distance measure between observed and simulated outputs.
    """
    return weight * np.sum(np.abs(y_obs - y_sim))
    return np.sum(((y_obs - y_sim) ** 2))

# Define a log-likelihood function
def log_likelihood(y_sim, y_obs):
    """
    Calculate the log-likelihood based on the distance function.
    """
    distance = distance_function(y_obs, y_sim, 10)
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
    #samples = list(set(tuple(row) for row in samples))
    return pd.DataFrame(samples, columns= column_names + ["ACTIVITY"])

def main():
    # Load ABM data
    data_path = "../../data/ARCADE/C-feature_15.0_metric_15-04032023.csv"
    data = pd.read_csv(data_path)
    data = data[data["COMPONENTS"] == 1]
    threshold = 0.2
    columns_to_drop = [col for col in data.columns if ((data[col] == np.inf) | (data[col] == -np.inf)).mean() >= threshold]
    data = data.drop(columns=columns_to_drop)

    # Extract inputs (theta) and outputs (y)
    input_feature_names = column_names #["NODES", "EDGES", "GRADIUS"]
    # input_feature_names = ["ACTIVITY"]
    predicted_output = ["ACTIVITY"]#, "GROWTH", "SYMMETRY"]
    input_features = data[input_feature_names].values
    
    y_sims = data[predicted_output].values

    # Observed value
    y_obs = [0.25]#, -10, 0]

    # Run MCMC
    n_iterations = 10000
    posterior_samples = mcmc(input_features, y_sims, y_obs, n_iterations, proposal_std=5.0)

    # Save posterior samples to a file
    posterior_samples.to_csv("posterior_samples_mcmc.csv", index=False)

    # Print summary of posterior samples
    print(f"Number of samples: {len(posterior_samples)}")
    print(posterior_samples.describe())
    # Plot the accepted samples activity
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    _, bins, patch = ax[0].hist(y_sims, bins=20)
    ax[0].set_title("Prior - Activity")
    ax[0].set_xlim([-1, 1])
    ax[0].set_xlabel("Activity")
    ax[0].set_ylabel("Number of samples")
    ax[1].hist(posterior_samples["ACTIVITY"], bins=bins)
    ax[1].set_title("Posterior - Activity (MCMC)")
    ax[1].set_xlim([-1, 1])
    ax[1].set_xlabel("Activity")
    ax[1].axvline(y_obs[0], color="red", linestyle="--", label="Target activity")
    ax[1].legend()

    pca = PCA(n_components=2)
    scaler = StandardScaler()
    features = scaler.fit_transform(input_features[:, 1:])
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(input_features[:, 0])
    reduced_features = pca.fit_transform(features)
    categories = label_encoder.classes_
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    unique_labels = np.unique(labels)
    cmap = plt.cm.viridis
    # drop duplicates
    posterior_samples = posterior_samples.drop_duplicates(subset=input_feature_names)
    posterior_reduced_features = pca.transform(scaler.transform(posterior_samples[input_feature_names].values[:, 1:]))
    posterior_labels = label_encoder.transform(posterior_samples[input_feature_names].values[:, 0])

    for i, label in enumerate(unique_labels):
        ax[2].scatter(reduced_features[labels == label, 0],
                      reduced_features[labels == label, 1], 
                      marker=markers[i % len(markers)],
                      label=f"{categories[label]}", 
                      facecolors='none',
                      edgecolors=cmap(i / len(unique_labels))
                      )
        ax[2].scatter(posterior_reduced_features[posterior_labels == label, 0], 
                      posterior_reduced_features[posterior_labels == label, 1],
                      marker=markers[i % len(markers)],
                      facecolors=cmap(i / len(unique_labels)),
                      edgecolors='none', alpha=0.8
                      )

    # Create custom legends
    handles1 = [plt.Line2D([0], [0], marker=markers[i % len(markers)], color='w', label=categories[label],
                           markerfacecolor='none', markeredgecolor=cmap(i / len(unique_labels))) 
                for i, label in enumerate(unique_labels)]
    handles2 = [plt.Line2D([0], [0], marker='o', color='w', label='Prior', markerfacecolor='none', markeredgecolor='k'),
                plt.Line2D([0], [0], marker='o', color='w', label='Posterior', markerfacecolor='k', markeredgecolor='none', alpha=0.5)]

    legend1 = ax[2].legend(handles=handles1, title="Vasculature type", loc='upper right')
    ax[2].add_artist(legend1)
    ax[2].legend(handles=handles2, title="Distribution", loc='lower right')
    ax[2].set_title("PCA - Vasculature distribution")
    ax[2].set_xlabel("PC1")
    ax[2].set_ylabel("PC2")
    plt.tight_layout()

    plt.savefig("posterior_mcmc.png")

if __name__ == "__main__":
    main()
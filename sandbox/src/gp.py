import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plot_parity_with_uncertainty(
    y_true_train, y_pred_train, y_std_train, 
    y_true_test, y_pred_test, y_std_test,
    xlim=(-3, 3), ylim=(-3, 3),
    train_r2=None, test_r2=None,
    filename="parity_plot_with_smooth_band.png"
):
    """
    Plots parity plots with uncertainty bands for training and test data.

    Parameters:
    - y_true_train: True values for the training data.
    - y_pred_train: Predicted values for the training data.
    - y_std_train: Predicted standard deviations for the training data.
    - y_true_test: True values for the test data.
    - y_pred_test: Predicted values for the test data.
    - y_std_test: Predicted standard deviations for the test data.
    - output_name: Name of the output variable being plotted (string).
    - xlim: Tuple specifying x-axis limits (default: (-2, 2)).
    - ylim: Tuple specifying y-axis limits (default: (-2, 2)).
    - filename: Name of the file to save the plot (default: "parity_plot_with_smooth_band.png").
    """
    # Sort data points for smooth connections
    sorted_indices_train = np.argsort(y_true_train[:, 0])
    y_true_train_sorted = y_true_train[sorted_indices_train, 0]
    y_pred_train_sorted = y_pred_train[sorted_indices_train, 0]
    y_std_train_sorted = y_std_train[sorted_indices_train, 0]

    sorted_indices_test = np.argsort(y_true_test[:, 0])
    y_true_test_sorted = y_true_test[sorted_indices_test, 0]
    y_pred_test_sorted = y_pred_test[sorted_indices_test, 0]
    y_std_test_sorted = y_std_test[sorted_indices_test, 0]

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Train data parity plot
    ax[0].plot(y_true_train_sorted, y_pred_train_sorted, label="Predicted", color="blue")
    ax[0].fill_between(
        y_true_train_sorted,
        y_pred_train_sorted - y_std_train_sorted,
        y_pred_train_sorted + y_std_train_sorted,
        color="blue",
        alpha=0.2,
        label="Uncertainty"
    )
    ax[0].scatter(y_true_train[:, 0], y_pred_train[:, 0], label="Train Data", color="blue", alpha=0.6)
    ax[0].plot(xlim, xlim, 'k--', label="Parity Line")
    ax[0].set_title(f"Parity Plot - Train Data (r2 = {train_r2:.3f})")
    ax[0].set_xlabel("True Values")
    ax[0].set_ylabel("Predicted Values")
    ax[0].legend()
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)

    # Test data parity plot
    ax[1].plot(y_true_test_sorted, y_pred_test_sorted, label="Predicted", color="green")
    ax[1].fill_between(
        y_true_test_sorted,
        y_pred_test_sorted - y_std_test_sorted,
        y_pred_test_sorted + y_std_test_sorted,
        color="green",
        alpha=0.2,
        label="Uncertainty"
    )
    ax[1].scatter(y_true_test[:, 0], y_pred_test[:, 0], label="Test Data", color="green", alpha=0.6)
    ax[1].plot(xlim, xlim, 'k--', label="Parity Line")
    ax[1].set_title(f"Parity Plot - Test Data (r2 = {test_r2:.3f})")
    ax[1].set_xlabel("True Values")
    ax[1].set_ylabel("Predicted Values")
    ax[1].legend()
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(ylim)

    # Final layout adjustments and save the figure
    plt.tight_layout()
    plt.savefig(filename)


OUTPUT_MAPPING = {"ACTIVITY": 0, "GROWTH": 1, "SYMMETRY": 2}

# Load data
data_path = "../../data/ARCADE/C-feature_0.0_metric_15-04032023.csv"
data = pd.read_csv(data_path)
data = data[data["KEY"] == "C_Lava"]
output_names = ["ACTIVITY", "GROWTH", "SYMMETRY"]
features = [
    "RADIUS", "LENGTH", "WALL", "SHEAR", "CIRCUM", "FLOW", 
    "NODES", "EDGES", "GRADIUS", "GDIAMETER", "AVG_ECCENTRICITY", 
    "AVG_SHORTEST_PATH", "AVG_IN_DEGREES", "AVG_OUT_DEGREES", 
    "AVG_DEGREE", "AVG_CLUSTERING", "AVG_CLOSENESS", 
    "AVG_BETWEENNESS", "AVG_CORENESS"
]
features = ["RADIUS"]
spatial_features = [
    "RADIUS", "LENGTH", "WALL", "SHEAR", "CIRCUM", "FLOW", 
    "NODES", "EDGES", "GRADIUS", "GDIAMETER", "AVG_ECCENTRICITY", 
    "AVG_SHORTEST_PATH", "AVG_IN_DEGREES", "AVG_OUT_DEGREES", 
    "AVG_DEGREE", "AVG_CLUSTERING", "AVG_CLOSENESS", 
    "AVG_BETWEENNESS", "AVG_CORENESS", "GRADIUS:FLOW", 
    "GDIAMETER:FLOW", "AVG_ECCENTRICITY:FLOW", "AVG_SHORTEST_PATH:FLOW", 
    "AVG_CLOSENESS:FLOW", "AVG_BETWEENNESS:FLOW", "GRADIUS:WALL", 
    "GDIAMETER:WALL", "AVG_ECCENTRICITY:WALL", "AVG_SHORTEST_PATH:WALL", 
    "AVG_CLOSENESS:WALL", "AVG_BETWEENNESS:WALL", "GRADIUS:SHEAR", 
    "GDIAMETER:SHEAR", "AVG_ECCENTRICITY:SHEAR", "AVG_SHORTEST_PATH:SHEAR", 
    "AVG_CLOSENESS:SHEAR", "AVG_BETWEENNESS:SHEAR", "GRADIUS:RADIUS", 
    "GDIAMETER:RADIUS", "AVG_ECCENTRICITY:RADIUS", "AVG_SHORTEST_PATH:RADIUS", 
    "AVG_CLOSENESS:RADIUS", "AVG_BETWEENNESS:RADIUS", "GRADIUS:PRESSURE_AVG", 
    "GDIAMETER:PRESSURE_AVG", "AVG_ECCENTRICITY:PRESSURE_AVG", 
    "AVG_SHORTEST_PATH:PRESSURE_AVG", "AVG_CLOSENESS:PRESSURE_AVG", 
    "AVG_BETWEENNESS:PRESSURE_AVG", "GRADIUS:PRESSURE_DELTA", 
    "GDIAMETER:PRESSURE_DELTA", "AVG_ECCENTRICITY:PRESSURE_DELTA", 
    "AVG_SHORTEST_PATH:PRESSURE_DELTA", "AVG_CLOSENESS:PRESSURE_DELTA", 
    "AVG_BETWEENNESS:PRESSURE_DELTA", "GRADIUS:OXYGEN_AVG", 
    "GDIAMETER:OXYGEN_AVG", "AVG_ECCENTRICITY:OXYGEN_AVG", 
    "AVG_SHORTEST_PATH:OXYGEN_AVG", "AVG_CLOSENESS:OXYGEN_AVG", 
    "AVG_BETWEENNESS:OXYGEN_AVG", "GRADIUS:OXYGEN_DELTA", 
    "GDIAMETER:OXYGEN_DELTA", "AVG_ECCENTRICITY:OXYGEN_DELTA", 
    "AVG_SHORTEST_PATH:OXYGEN_DELTA", "AVG_CLOSENESS:OXYGEN_DELTA", 
    "AVG_ECCENTRICITY_WEIGHTED", 
    "AVG_CLOSENESS_WEIGHTED", "AVG_CORENESS_WEIGHTED", 
    "AVG_BETWEENNESS_WEIGHTED", "AVG_OUT_DEGREES_WEIGHTED", 
    "AVG_IN_DEGREES_WEIGHTED", "AVG_DEGREE_WEIGHTED", 
    "GRADIUS:INVERSE_DISTANCE", "GDIAMETER:INVERSE_DISTANCE", 
    "AVG_ECCENTRICITY:INVERSE_DISTANCE", "AVG_SHORTEST_PATH:INVERSE_DISTANCE", 
    "AVG_CLOSENESS:INVERSE_DISTANCE", "AVG_BETWEENNESS:INVERSE_DISTANCE"
]
selected_features_indices = [2,0,17,6,7,3,18,4,11,15]
features = np.array(features)#[selected_features_indices]
selected_features = features

#features = spatial_features

kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1, length_scale_bounds=(1e-2, 1e2), nu=1.5) #+ WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-4, 1e-1))
train = True

feature_selection_results = {}
for iteration in range(1):#(len(features)):
    print("="*10 + f"Random Feature Set {iteration+1}" + "="*10)
    #selected_features = random.sample(features, random.randint(2, 3))
    #X = data[selected_features].values
    selected_features = np.array(features)#[:iteration+1]
    X = data[selected_features].values  # Inputs
    y = data[output_names].values  # Outputs


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("X_train, X_test, y_train, y_test")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.transform(y_test)
    if train:
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=3e-6)
        gp.fit(X_train, y_train)
    else:
        gp = joblib.load('gp.pkl')

    # joblib.dump(gp, 'gp.pkl')

    y_pred, y_pred_std = gp.predict(X_test, return_std=True)
    y_pred_train, y_pred_std_train = gp.predict(X_train, return_std=True)
    # Plot GP prediction function with uncertainty
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_pca = False
    if plot_pca:
        pca = PCA(n_components=1)  # Focus on PC1 for plotting
        pca_X_train = pca.fit_transform(X_train)
        pca_X_test = pca.transform(X_test)
        # Generate uniform points in the PC1 space
        pc1_min, pc1_max = pca_X_train.min(), pca_X_train.max()
        pc1_uniform = np.linspace(pc1_min, pc1_max, 50).reshape(-1, 1)
        # Map uniform points back to the original feature space
        X_uniform = pca.inverse_transform(pc1_uniform)
        # Get GP predictions (mean and standard deviation) for the uniform points
        y_p, y_std = gp.predict(X_uniform, return_std=True)

        # Plot GP prediction with uncertainty
        # Scatter plot for training data in PC1
        print(pca_X_train.shape, y_train.shape)
        ax.scatter(pca_X_train[:, 0], y_train[:, 0], label="Train Data", color="blue", alpha=0.6)
        ax.scatter(pca_X_test[:, 0], y_test[:, 0], label="Test Data", color="green", alpha=0.6)

        # GP prediction mean
        ax.scatter(pc1_uniform, y_p[:, 0], label="GP Prediction", color="red", linewidth=2)
        ax.plot(pc1_uniform, y_p[:, 0], label="GP Prediction", color="red", linewidth=2)
        # Customize the plot
        ax.set_title("GP Prediction with Uncertainty")
        ax.set_xlabel("Principal Component 1 (PC1)")
        ax.set_ylabel("Prediction")
    else:
        x = np.linspace(-3, 3, 1000).reshape(-1, 1)
        y_p = gp.predict(x)
        ax.scatter(X_train, y_train[:, 0], label="Train")
        ax.scatter(X_test, y_test[:, 0], label="Test")
        ax.plot(x, y_p[:, 0], label="Prediction", color="red")
        ax.set_title("GP Prediction")
        ax.set_xlabel("RADIUS")
        ax.set_ylabel("Prediction")

    ax.legend()

    plt.tight_layout()
    plt.savefig("gp.png")
    """
    y_pred = scaler.inverse_transform(y_pred)
    y_pred_std = scaler.inverse_transform(y_pred_std)
    y_pred_train = scaler.inverse_transform(y_pred_train)
    y_pred_std_train = scaler.inverse_transform(y_pred_std_train)
    y_train = scaler.inverse_transform(y_train)
    y_test = scaler.inverse_transform(y_test)
    """
    for i, name in enumerate(output_names[:1]):
        r2_train = r2_score(y_train[:, i], y_pred_train[:, i])
        r_train = np.corrcoef(y_train[:, i], y_pred_train[:, i])[0, 1]
        mse_train = mean_squared_error(y_train[:, i], y_pred_train[:, i])
        r2_test = r2_score(y_test[:, i], y_pred[:, i])
        r_test = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
        mse_test = mean_squared_error(y_test[:, i], y_pred[:, i])
        # Save results
        feature_selection_results[f"Feature Set {iteration+1}"] = {
            "Selected Features": selected_features,
            "R² Train": f"{r2_train:.3f}",
            "R² Test": f"{r2_test:.3f}",
            "MSE Train": f"{mse_train:.3f}",
            "MSE Test": f"{mse_test:.3f}",
        }
    plot_parity_with_uncertainty(y_train,
                                 y_pred_train,
                                 y_pred_std_train,
                                 y_test,
                                 y_pred,
                                 y_pred_std,
                                 train_r2=r2_train,
                                 test_r2=r2_test,
                                 )
    asd()
    # Plot a parity plot for train and test data
    output_name = output_names[0]
    output_index = OUTPUT_MAPPING[output_name]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(y_train[:, output_index], y_pred_train[:, output_index], label="Train")
    ax.scatter(y_test[:, output_index], y_pred[:, output_index], label="Test")
    ax.set_title(f"{output_name} (R^2 Train: {r2_train:.3f}, R^2 Test: {r2_test:.3f})")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.legend()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.tight_layout()

    # Save the plot
    selected_features_combined = "_".join(selected_features)
    plt.savefig(f"parity_plot_{selected_features_combined}.png")

results_df = pd.DataFrame.from_dict(feature_selection_results, orient="index")
results_df.sort_values(by="R² Test", ascending=False, inplace=True)

# Save results to a CSV
results_df.to_csv("feature_selection_results.csv", index=False)
print(results_df.head(1))

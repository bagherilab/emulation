import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def clean_data(full_data, response):
    """Handle missing or non-numeric data"""

    # Remove rows with multiple components
    full_data = full_data[full_data["COMPONENTS"] == 1]
    full_data.reset_index(drop=True, inplace=True)

    # Remove response rows with bad values
    full_data = full_data.loc[~full_data[response].isin([np.nan, np.inf, -np.inf])]
    full_data.reset_index(drop=True, inplace=True)

    # Removed features columns with bad values
    numeric_cols = full_data.select_dtypes(include=[np.number]).columns
    full_data = full_data.loc[
        :, ~(np.isnan(full_data[numeric_cols]).any(axis=0) | np.isinf(full_data[numeric_cols])).any(axis=0)
    ]
    return full_data

OUTPUT_MAPPING = {"ACTIVITY": 0, "GROWTH": 1, "SYMMETRY": 2}

# Load data
data_path = "../../data/ARCADE/C-feature_0.0_metric_15-04032023.csv"
data = pd.read_csv(data_path)
output_names = ["ACTIVITY", "GROWTH", "SYMMETRY"]
features = [
    "RADIUS", "LENGTH", "WALL", "SHEAR", "CIRCUM", "FLOW", 
    "NODES", "EDGES", "GRADIUS", "GDIAMETER", "AVG_ECCENTRICITY", 
    "AVG_SHORTEST_PATH", "AVG_IN_DEGREES", "AVG_OUT_DEGREES", 
    "AVG_DEGREE", "AVG_CLUSTERING", "AVG_CLOSENESS", 
    "AVG_BETWEENNESS", "AVG_CORENESS"
]
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
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1, length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-4, 1e-1))
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
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.transform(y_test)
    
    if train:
        mlp = MLPRegressor(hidden_layer_sizes=(5, 10),
                           activation="logistic",
                           alpha=0.316,
                           solver="lbfgs",
                           max_iter=1000,
                           random_state=42)
        mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    y_pred_train = mlp.predict(X_train)
    # Convert back to original scale
    """
    y_pred = scaler.inverse_transform(y_pred)
    y_pred_train = scaler.inverse_transform(y_pred_train)
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
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(xlim, xlim, 'k--', label="Parity Line")
    plt.tight_layout()

    # Save the plot
    selected_features_combined = "_".join(selected_features)
    plt.savefig(f"parity_plot.png")#_{selected_features_combined}.png")

results_df = pd.DataFrame.from_dict(feature_selection_results, orient="index")
results_df.sort_values(by="R² Test", ascending=False, inplace=True)

# Save results to a CSV
#results_df.to_csv("feature_selection_results.csv", index=False)
print(results_df.head(1))

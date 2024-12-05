import random
import shap
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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
selected_features_indices = [0, 2,3,4,6,7,11,15,17,18]
features = np.array(features)#[selected_features_indices]

kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1, length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-4, 1e-1))

# Load the data
X = data[features].values  # Use all features
y = data[output_names].values  # Outputs
y = y[:, OUTPUT_MAPPING["ACTIVITY"]]  # Choose the output to predict
y = y.reshape(-1, 1)  # Reshape to 2D array

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)
# Train Gaussian Process model with all features
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-3)
gp.fit(X_train, y_train)
background_sample_size = 30  # Choose an appropriate size for your data and model
background_data = shap.sample(X_train, background_sample_size)  # Random sampling
# SHAP Analysis
explainer = shap.KernelExplainer(gp.predict, background_data)  # SHAP Kernel Explainer for GP model
shap_values = explainer.shap_values(X_test[:100])  # Explain first 100 test samples

# Visualize Global Feature Importance
shap.summary_plot(shap_values, X_test[:100], feature_names=features)
# save figure
plt.savefig("shap_summary_plot.png")

# Optionally, print the mean absolute SHAP values for ranking
feature_importance = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({"Feature": features, "Mean SHAP Value": feature_importance})
importance_df.sort_values(by="Mean SHAP Value", ascending=False, inplace=True)

# Display the most important features
print("Most Important Features:")
print(importance_df.head(10))

# Save feature importance to CSV
importance_df.to_csv("shap_feature_importance.csv", index=False)

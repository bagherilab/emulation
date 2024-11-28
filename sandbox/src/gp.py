import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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
output_names = ["ACTIVITY"]#, "GROWTH", "SYMMETRY"]
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
features = spatial_features
#features = ["NODES", "EDGES"]
# Input features and outputs
# Remove rows with NaN or infinite values

data = clean_data(data, output_names[0])

X = data[features].values  # Inputs
y = data[output_names].values  # Outputs


# Step 1: Split data into 75% training and 25% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 2: Split the training set into 75% training and 25% validation
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the GP kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

# Train a separate GP for each output using cross-validation

# Step 3: Perform 5-fold cross-validation on the training set
train = True
if train:
    kf = None
    if kf is not None:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        gps = []
        cv_scores = []

        for i in range(y_train_final.shape[1]):  # Train a GP for each output (ACTIVITY, GROWTH, SYMMETRY)
            fold_scores = []
            for train_idx, val_idx in kf.split(X_train_final):
                # Split the data into training and validation sets for this fold
                X_train_fold, X_val_fold = X_train_final[train_idx], X_train_final[val_idx]
                y_train_fold, y_val_fold = y_train_final[train_idx, i], y_train_final[val_idx, i]
                
                # Train the GP on the current fold
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
                gp.fit(X_train_fold, y_train_fold)
                
                # Evaluate on the validation fold
                y_val_pred, y_val_std = gp.predict(X_val_fold, return_std=True)
                fold_score = np.mean((y_val_pred - y_val_fold) ** 2)  # Mean Squared Error
                fold_scores.append(fold_score)
            
            # Store the GP model and cross-validation score
            gps.append(gp)
            cv_scores.append(np.mean(fold_scores))
        # Print cross-validation scores
        for name, score in zip(output_names, cv_scores):
            print(f"Cross-validation MSE for {name}: {score:.4f}")
    else:
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=2e-2)
        gp.fit(X_train, y_train)
        gps = [gp]
else:
    gps = [joblib.load('gp.pkl')]


# Step 4: Final evaluation on the test set
# Select the GP model with the lowest cross-validation score
gp = gps[0]  # GP model for "ACTIVITY"
# save the model
joblib.dump(gp, 'gp.pkl')

y_pred, y_pred_std = gp.predict(X_test, return_std=True)
y_pred_train, y_pred_std_train = gp.predict(X_train, return_std=True)
for i, name in enumerate(output_names[:1]):
    r2_train = r2_score(y_train[:, i], y_pred_train[:, i])
    r_train = np.corrcoef(y_train[:, i], y_pred_train[:, i])[0, 1]
    mse_train = mean_squared_error(y_train[:, i], y_pred_train[:, i])
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    r = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
    mse = mean_squared_error(y_test[:, i], y_pred[:, i])

    print("="*10 + "TRAIN" + "="*10)
    print(f"R-squared for {name}: {r2_train:.4f}")
    print(f"R for {name}: {r_train:.4f}")
    print(f"MSE for {name}: {mse_train:.4f}")
    print(f"Std for {name}: {y_pred_std_train[i]}")
    print("="*10 + "TEST" + "="*10)
    print(f"R-squared for {name}: {r2}")
    print(f"R for {name}: {r:.4f}")
    print(f"MSE for {name}: {mse:.4f}")
    print(f"Std for {name}: {y_pred_std[i]}")

# Create a DataFrame with required columns for three outputs
results_df = pd.DataFrame({
    "ACTIVITY_true": y_test[:, 0],
    "ACTIVITY_pred": y_pred[:, 0],
    "ACTIVITY_std": y_pred_std[:, 0],
    "GROWTH_true": y_test[:, 1],
    "GROWTH_pred": y_pred[:, 1],
    "GROWTH_std": y_pred_std[:, 1],
    "SYMMETRY_true": y_test[:, 2],
    "SYMMETRY_pred": y_pred[:, 2],
    "SYMMETRY_std": y_pred_std[:, 2],
})
# Save to CSV
results_df.to_csv("activity_predictions.csv", index=False)
# print(results_df.head())

# Plot a parity plot for train and test data
output_name = output_names[0]
output_index = OUTPUT_MAPPING[output_name]
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(y_train[:, output_index], gp.predict(X_train)[:, output_index], label="Train")
ax.scatter(y_test[:, output_index], y_pred[:, output_index], label="Test")
ax.set_title(output_name)
ax.set_xlabel("True")
ax.set_ylabel("Predicted")
ax.legend()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.tight_layout()

# Save the plot
plt.savefig("parity_plot.png")

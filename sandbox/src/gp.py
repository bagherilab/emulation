import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score


# Load data
data_path = "../../data/ARCADE/C-feature_0.0_metric_15-04032023.csv"
data = pd.read_csv(data_path)
output_names = ["ACTIVITY", "GROWTH", "SYMMETRY"]

# Input features and outputs
X = data[["NODES", "EDGES", "GRADIUS"]].values  # Inputs
y = data[output_names].values  # Outputs


# Step 1: Split data into 75% training and 25% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 2: Split the training set into 75% training and 25% validation
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Step 3: Perform 5-fold cross-validation on the training set
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the GP kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

# Train a separate GP for each output using cross-validation
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

# Step 4: Final evaluation on the test set
# Final evaluation for the "ACTIVITY" output
activity_gp = gps[0]  # GP model for "ACTIVITY"
y_test_activity = y_test[:, 0]  # True test values for "ACTIVITY"
y_pred_activity, y_pred_std = activity_gp.predict(X_test, return_std=True)  # Predictions for "ACTIVITY"

# Calculate MSE and R-squared
mse_activity = mean_squared_error(y_test_activity, y_pred_activity)
r2_activity = r2_score(y_test_activity, y_pred_activity)

# Print R-squared value
print(f"R-squared for ACTIVITY: {r2_activity:.4f}")
print(f"MSE for ACTIVITY: {mse_activity:.4f}")


# Create a DataFrame with required columns
results_df = pd.DataFrame({
    "test_activity": y_test_activity,
    "pred_activity": y_pred_activity,
})

# Save to CSV
results_df.to_csv("activity_predictions.csv", index=False)

# Print the first few rows of the results
print("\nSaved predictions to 'activity_predictions.csv':")
print(results_df.head())

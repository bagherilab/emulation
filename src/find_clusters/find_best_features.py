import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl

from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from combine_temporal_data import load_data
import numpy as np

def classify_and_visualize(data,
                           time_point,
                           features,
                           label_column):
    # Encode labels as integers
    data = data.copy()
    data_at_time = data[data['TIME'] == time_point].copy()  # Use .copy() to avoid the warning
    label_encoder = LabelEncoder()
    data_at_time[label_column] = label_encoder.fit_transform(data_at_time[label_column])
    
    X = data_at_time[features]
    y = data_at_time[label_column]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Feature selection (ANOVA F-value)
    selector = SelectKBest(score_func=f_classif, k='all')
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Rank features by importance
    feature_scores = pd.DataFrame({
        "Feature": features,
        "Score": selector.scores_
    }).sort_values(by="Score", ascending=False)
    print("Feature Importance Rankings:")
    print(feature_scores)
    # Train an SVM classifier
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train_selected, y_train)
    y_pred = svm.predict(X_test_selected)
    
    # Evaluate performance
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    
    return feature_scores


def visualize_features_response(data, time_point, features, response_name, label_column):
    # Filter data for the specified time point
    data_at_time = data[data['TIME'] == time_point].copy()
    data_at_time = data_at_time[data_at_time["COMPONENTS"] == 1]
    threshold = 0.2
    columns_to_drop = [col for col in data_at_time.columns if ((data_at_time[col] == np.inf) | (data_at_time[col] == -np.inf)).mean() >= threshold]
    data_at_time = data_at_time.drop(columns=columns_to_drop)
    label_encoder = LabelEncoder()
    data_at_time[label_column] = label_encoder.fit_transform(data_at_time[label_column])
    data_at_time = data_at_time.replace([float('inf'), float('-inf')], float('nan')).dropna(axis=0)
    # Encode labels and prepare data
    X = data_at_time[features]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = data_at_time[label_column]
    categories = label_encoder.classes_
    
    # Define a consistent colormap
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=min(y), vmax=max(y))  # Normalize the label encoding
    category_colors = {i: cmap(norm(i)) for i in range(len(categories))}  # Map encoded labels to colors
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(X)

    pc1_loadings = pca.components_[0]  # First principal component
    feature_importance = pd.DataFrame({
        "Feature": features,
        "Loading": pc1_loadings,
        "Absolute Loading": abs(pc1_loadings)
    })
    
    # Rank features by absolute loading
    feature_importance = feature_importance.sort_values(by="Absolute Loading", ascending=False)
    print("Feature Importance Rankings:")
    print(feature_importance)

    #reduced_features[:, 1] *= 0
    print("PCA Explained Variance Ratio:")
    print(pca.explained_variance_ratio_)
    # Create a heatmap for feature correlation
    plt.figure(figsize=(12, 10))
    sns.heatmap(data[features].corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title("Feature Correlation Heatmap")
    plt.savefig("feature_correlation_heatmap.png")
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})
    # PCA scatter plot
    scatter = axes[0].scatter(reduced_features[:, 0],
                              reduced_features[:, 1],
                              c=y,
                              cmap=cmap,
                              s=50)
    
    # Add custom colorbar with string labels
    colorbar = fig.colorbar(scatter, ax=axes[0], label="Category")
    colorbar_ticks = range(len(categories))
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels(categories)  # Use string labels for colorbar
    
    # Add legend to PCA plot
    handles, _ = scatter.legend_elements()
    axes[0].legend(handles, categories, title="Categories")
    axes[0].set_xlabel("PCA Component 1")
    axes[0].set_ylabel("PCA Component 2")
    axes[0].set_title("PCA Visualization of Features")
    
    # Violin plot for response distribution
    sns.boxplot(x=label_column, y=response_name, data=data_at_time, ax=axes[1], palette=category_colors)
    sns.swarmplot(x=label_column, y=response_name, data=data_at_time, ax=axes[1], color='black')
    # Set custom x-ticks using category names
    axes[1].set_xticks(range(len(categories)))  # Set the positions of the ticks
    axes[1].set_xticklabels(categories)  # Set the category names as tick labels
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel(response_name)
    axes[1].set_title(f"{response_name} Distribution for TIME={time_point}")
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig("combined_visualization.png")
    plt.show()

def visualize_multiple_vasculatures_over_time(data, vasculature_types, features, label_column):
    # Filter data for the selected vasculature types
    data_filtered = data[data[label_column].isin(vasculature_types)].copy()

    # Remove columns with excessive inf/-inf values
    threshold = 0.2
    columns_to_drop = [col for col in data_filtered.columns if ((data_filtered[col] == np.inf) | (data_filtered[col] == -np.inf)).mean() >= threshold]
    data_filtered = data_filtered.drop(columns=columns_to_drop)

    # Replace inf/-inf with NaN and drop rows with NaN
    data_filtered = data_filtered.replace([float('inf'), float('-inf')], float('nan')).dropna(axis=0)

    # Reset the index
    data_filtered = data_filtered.reset_index(drop=True)

    # Standardize features
    X = data_filtered[features]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(X)

    # PCA Explained Variance
    print("PCA Explained Variance Ratio:")
    print(pca.explained_variance_ratio_)

    # Feature importance (loadings for PC1)
    pc1_loadings = pca.components_[0]
    feature_importance = pd.DataFrame({
        "Feature": features,
        "Loading": pc1_loadings,
        "Absolute Loading": abs(pc1_loadings)
    }).sort_values(by="Absolute Loading", ascending=False)
    print("Feature Importance Rankings:")
    print(feature_importance)

    # Define colormap for time points and markers for vasculature types
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(data_filtered['TIME']), vmax=max(data_filtered['TIME']))
    markers = ['o', 's', '^', 'D', 'P', 'X']  # Add more markers if needed

    # Plot PCA scatter for multiple vasculature types
    plt.figure(figsize=(10, 7))
    for i, vasculature in enumerate(vasculature_types):
        subset = data_filtered[data_filtered[label_column] == vasculature]
        reduced_subset = reduced_features[subset.index]  # Indices now align after reset_index
        plt.scatter(
            reduced_subset[:, 0], reduced_subset[:, 1],
            c=subset['TIME'], cmap=cmap, norm=norm, s=50,
            marker=markers[i % len(markers)], label=vasculature
        )

    # Add colorbar and legend
    colorbar = plt.colorbar(label="Time")
    plt.legend(title="Vasculature Type")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA Visualization of Multiple Vasculature Types Over Time")
    plt.savefig("pca_multiple_vasculatures_over_time.png")
    plt.show()

    # Feature Correlation Heatmap for all selected vasculatures
    plt.figure(figsize=(12, 10))
    sns.heatmap(data_filtered[features].corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title("Feature Correlation Heatmap")
    plt.savefig("feature_correlation_multiple_vasculatures.png")
    plt.show()



def main():
    # Example usage
    label_column = "KEY"
    features = [
        "RADIUS", "LENGTH", "WALL", "SHEAR", "CIRCUM", "FLOW", 
        "NODES", "EDGES", "GRADIUS", "GDIAMETER", "AVG_ECCENTRICITY", 
        "AVG_SHORTEST_PATH", "AVG_IN_DEGREES", "AVG_OUT_DEGREES", 
        "AVG_DEGREE", "AVG_CLUSTERING", "AVG_CLOSENESS", 
        "AVG_BETWEENNESS", "AVG_CORENESS"
    ]
    #features = [
#
#    ]
    # Define the pattern to locate the files and filter suffix
    data_path_pattern = os.path.join(os.path.dirname(__file__), "../../data/ARCADE/C-feature_*.csv")
    suffix_filter = "_15-04032023.csv"  # Specify the suffix to filter files
    data = load_data(data_path_pattern, suffix_filter)
    data = data[(data['LAYOUT'] == 'Savav') | (data['LAYOUT'] == 'Lav')]
    time_point = 15.0
    response_name = "ACTIVITY"
    visualize_features_response(data, time_point, features, response_name, label_column)
    vasculature_types = ["C_Savav", "C_Lav"]
    #visualize_multiple_vasculatures_over_time(data, vasculature_types, features, label_column)
if __name__ == "__main__":
    main()
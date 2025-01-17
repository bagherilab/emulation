import os
from glob import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
import seaborn as sns
from combine_temporal_data import load_data

def cluster_analysis_with_ground_truth(data, time_point, features, label_column):
    # Filter data for the specified time point
    data_at_time = data[data['TIME'] == time_point].copy()  # Use .copy() to avoid the warning
    
    # Select only the features for clustering
    feature_data = data_at_time[features]
    
    # Encode string ground truth labels to integers
    label_encoder = LabelEncoder()
    ground_truth_labels = label_encoder.fit_transform(data_at_time[label_column])  # Encode string labels to integers
    
    # Normalize the features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_data)
    
    optimal_clusters = len(set(ground_truth_labels))  # Use the number of unique ground truth categories
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(normalized_features)
    
    data_at_time.loc[:, 'CLUSTER'] = cluster_labels
    
    # Compare clustering with ground truth
    ari = adjusted_rand_score(ground_truth_labels, cluster_labels)
    nmi = normalized_mutual_info_score(ground_truth_labels, cluster_labels)
    print(f"Adjusted Rand Index (ARI): {ari}")
    print(f"Normalized Mutual Information (NMI): {nmi}")
    
    # Confusion Matrix
    cm = confusion_matrix(ground_truth_labels, cluster_labels)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(optimal_clusters), yticklabels=label_encoder.classes_)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.5, cm[i, j], ha='center', va='center', color='black')
    plt.xlabel("Predicted Clusters")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    
    # Optional PCA visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(normalized_features)
    plt.figure(figsize=(8, 5))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=ground_truth_labels, cmap='viridis', s=50, label='Ground Truth')
    #plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='plasma', s=20, label='Predicted Clusters', alpha=0.5)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"Clusters vs Ground Truth for TIME={time_point}")
    plt.legend()
    plt.savefig("pca_clusters.png")
    
    return data_at_time

def main():
    features = [
        "RADIUS", "LENGTH", "WALL", "SHEAR", "CIRCUM", "FLOW", 
        "NODES", "EDGES", "GRADIUS", "GDIAMETER", "AVG_ECCENTRICITY", 
        "AVG_SHORTEST_PATH", "AVG_IN_DEGREES", "AVG_OUT_DEGREES", 
        "AVG_DEGREE", "AVG_CLUSTERING", "AVG_CLOSENESS", 
        "AVG_BETWEENNESS", "AVG_CORENESS"
    ]
    label_column = "KEY"
    features = [
        "RADIUS", "LENGTH", "WALL", "SHEAR", "CIRCUM", "FLOW", "NODES", "EDGES"]
    # Define the pattern to locate the files and filter suffix
    data_path_pattern = os.path.join(os.path.dirname(__file__), "../../data/ARCADE/C-feature_*.csv")
    suffix_filter = "_15-04032023.csv"  # Specify the suffix to filter files
    data = load_data(data_path_pattern, suffix_filter)
    # Print number of rows
    data = data[(data['LAYOUT'] == 'Savav') | (data['LAYOUT'] == 'Lav')]
    # add graph density = nodes/edges
    data['DENSITY'] = data['NODES'] / data['EDGES']
    # print mean density of Lav and Savav
    print(data.groupby('LAYOUT')['DENSITY'].agg(['mean', 'std']))
    print(data.groupby('LAYOUT')['NODES'].agg(['mean', 'std']))
    print(data.groupby('LAYOUT')['EDGES'].agg(['mean', 'std']))
    print(data.head())  # Show the first few rows of the combined DataFrame
    #print(data['TIME'].unique())  # Display the unique time points
    # Analyze clusters for TIME=0.0 with ground truth comparison
    clustered_data = cluster_analysis_with_ground_truth(data, time_point=0.0, features=features, label_column=label_column)
    #print(clustered_data.head())  # View the clustered data


if __name__ == '__main__':
    main()

from sklearn.datasets import make_regression
import pandas as pd

n_features = 12
n_targets = 4
features, target = make_regression(
    n_samples=200,
    n_features=n_features,
    n_informative=8,
    n_targets=n_targets,
    random_state=42,
)

feature_columns = [f"feature_{x}" for x in range(n_features)]
df = pd.DataFrame(features, columns=feature_columns)
target_columns = [f"target_{y}" for y in range(n_targets)]
target_df = pd.DataFrame(target, columns=target_columns)

df = pd.concat([df, target_df.reindex(df.index)], axis=1).reset_index()
df.to_csv("toy_data.csv", index=False)

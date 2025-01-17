import os
import pandas as pd
import re
from glob import glob


def load_data(file_pattern, suffix_filter):
    all_data = []
    # Find all files matching the pattern
    files = glob(file_pattern)
    for file in files:
        # Skip files that don't match the suffix_filter
        if not file.endswith(suffix_filter):
            continue
        
        # Extract the time point from the filename
        time_match = re.search(r'C-feature_(\d+\.\d+)_metric', os.path.basename(file))
        if time_match:
            time_point = float(time_match.group(1))  # Extract and convert the time point
        else:
            raise ValueError(f"Time point not found in {file}")
        
        # Load the data and add the TIME column
        data = pd.read_csv(file)
        data['TIME'] = time_point  # Add the TIME column
        all_data.append(data)
    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)
    # Sort the data by TIME
    combined_data.sort_values(by='TIME', inplace=True)
    return combined_data

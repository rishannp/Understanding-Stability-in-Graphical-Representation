import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Define data for each subject in a flat dictionary format with fold pairs
subjects_data = {
    "S1": [0.3974, 0.4359, 0.5000, 0.3590, 0.4615, 0.5238, 0.5385, 0.5897, 0.5000, 0.5000, 0.5385],
    "S2": [0.7024, 0.6071, 0.5000, 0.6548, 0.6429, 0.5444, 0.6667, 0.5238, 0.5667, 0.3690, 0.4524],
    "S5": [0.5513, 0.6026, 0.6341, 0.5769, 0.5256, 0.7317, 0.5128, 0.4872, 0.5488, 0.5256, 0.6538],
    "S9": [0.6538, 0.5385, 0.5488, 0.6410, 0.5256, 0.6098, 0.5513, 0.6026, 0.5976, 0.5513, 0.6282],
    "S21": [0.6538, 0.6154, 0.6190, 0.5769, 0.6026, 0.6190, 0.6154, 0.5897, 0.5595, 0.6410, 0.6154],
    "S31": [0.7051, 0.6410, 0.7143, 0.7436, 0.7051, 0.7619, 0.6667, 0.7179, 0.7738, 0.7564, 0.7821],
    "S34": [0.9189, 0.9730, 0.9737, 0.9324, 0.9865, 0.9605, 0.9324, 0.9865, 0.9079, 0.9730, 0.9595],
    "S39": [0.8378, 0.6892, 0.8289, 0.9189, 0.8649, 0.8816, 0.8378, 0.8919, 0.8947, 0.8649, 0.8649]
}

# Define fold pairs for the x-axis
fold_pairs = ["1v2", "1v3", "1v4", "2v1", "2v3", "2v4", "3v1", "3v2", "3v4", "4v1", "4v2"]

# Convert data to DataFrame for plotting
df = pd.DataFrame(subjects_data, index=fold_pairs).T

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df, annot=True, fmt=".4f", cmap="YlGnBu", vmin=0.3, vmax=1.0, cbar=True)
plt.title("Accuracy Heatmap Across Subjects and Train-Test Pairs")
plt.xlabel("Train-Test Pair")
plt.ylabel("Subject")
plt.show()

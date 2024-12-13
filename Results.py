import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data for the various models (same structure as before)
data = {
    "Patient": [1, 2, 5, 9, 21, 31, 34, 39],
    "BP+LDA_1v2": [39.74, 70.24, 55.13, 65.38, 65.38, 70.51, 91.89, 83.78],
    "BP+LDA_1v3": [43.59, 60.71, 60.26, 53.85, 61.54, 64.10, 97.30, 68.92],
    "BP+LDA_1v4": [50.00, 50.00, 63.41, 54.88, 61.90, 71.43, 97.37, 82.89],
    "CSP+SVM_1v2": [51.28, 55.95, 51.28, 82.05, 55.13, 67.95, 50.00, 47.30],
    "CSP+SVM_1v3": [50.00, 54.76, 50.00, 67.95, 57.69, 57.69, 50.00, 47.30],
    "CSP+SVM_1v4": [47.62, 54.44, 50.00, 68.29, 50.00, 55.95, 59.21, 47.37],
    "CFC_Mu_Beta_1v2": [66.25, 68.60, 63.29, 73.42, 57.50, 68.75, 58.67, 66.67],
    "CFC_Mu_Beta_1v3": [62.50, 65.12, 64.56, 67.09, 60.00, 66.25, 60.00, 66.67],
    "CFC_Mu_Beta_1v4": [63.75, 61.63, 65.82, 58.23, 58.75, 60.00, 60.00, 65.33],
    "MSC_1v2": [61.25, 70.93, 62.03, 72.15, 65.00, 65.00, 64.00, 76.00],
    "MSC_1v3": [58.75, 63.95, 70.89, 63.29, 63.75, 66.25, 53.33, 64.00],
    "MSC_1v4": [52.50, 60.47, 65.82, 64.56, 62.50, 66.25, 57.33, 65.33],
    "PLV_1v2": [60.00, 69.77, 58.23, 74.68, 60.00, 67.50, 64.00, 68.00],
    "PLV_1v3": [56.25, 65.12, 64.56, 54.43, 68.75, 68.75, 69.33, 64.00],
    "PLV_1v4": [56.25, 60.47, 64.56, 63.29, 58.75, 60.00, 62.67, 65.33],
    "CMI_1v2": [60.00, 70.93, 60.76, 68.35, 70.00, 66.25, 68.00, 61.33],
    "CMI_1v3": [58.75, 67.44, 70.89, 56.96, 63.75, 60.00, 60.00, 69.33],
    "CMI_1v4": [66.25, 68.60, 65.82, 63.29, 65.00, 66.25, 61.33, 72.00],
    "EEGNet_1v2": [61.54, 53.66, 62.34, 64.94, 55.13, 50.00, 71.23, 54.79],
    "EEGNet_1v3": [56.41, 53.66, 55.84, 66.23, 60.26, 61.84, 54.79, 53.42],
    "EEGNet_1v4": [56.41, 58.54, 63.64, 53.25, 53.85, 57.89, 60.27, 63.01],
    "DeepConvNet_1v2": [62.82, 56.10, 57.14, 67.53, 57.69, 53.95, 60.27, 56.16],
    "DeepConvNet_1v3": [52.56, 63.41, 63.64, 63.64, 58.97, 50.00, 50.68, 49.32],
    "DeepConvNet_1v4": [55.13, 58.54, 59.74, 61.04, 53.85, 52.63, 54.79, 64.38]
}

# Create the dataframe
df = pd.DataFrame(data)

# Set Patient as the index
df.set_index('Patient', inplace=True)

# Create a dictionary for model titles
model_titles = {
    "BP+LDA": "Band Power with LDA",
    "CSP+SVM": "CSP with SVM",
    "CFC_Mu_Beta": "CFC (Mu and Beta)",
    "MSC": "MSC",
    "PLV": "PLV",
    "CMI": "CMI",
    "EEGNet": "EEGNet",
    "DeepConvNet": "DeepConvNet"
}

# List of models
models = ["BP+LDA", "CSP+SVM", "CFC_Mu_Beta", "MSC", "PLV", "CMI", "EEGNet", "DeepConvNet"]

# Create each figure for each model
for model in models:
    # Extract data for the current model
    model_data = df[[f"{model}_1v2", f"{model}_1v3", f"{model}_1v4"]]
    
    # Set up the plot
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Plot the heatmap
    sns.heatmap(model_data, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, vmin=0, vmax=100)
    
    # Formatting the plot
    plt.title(f'Accuracy Heatmap for {model_titles[model]}')
    plt.xlabel('Fold Pairs')
    plt.ylabel('Subjects')
    plt.xticks(ticks=np.arange(3) + 0.5, labels=["1v2", "1v3", "1v4"])
    plt.yticks(ticks=np.arange(len(df.index)) + 0.5, labels=df.index, rotation=0)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    

# Create the dataframe
df = pd.DataFrame(data)

# Set up models and fold pairs
models = ["BP+LDA", "CSP+SVM", "CFC_Mu_Beta", "MSC", "PLV", "CMI", "EEGNet", "DeepConvNet"]
fold_pairs = ["1v2", "1v3", "1v4"]

# Create line graph per subject
for subject in df.index:
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot each model's data
    for model in models:
        accuracies = [df.loc[subject, f"{model}_{pair}"] for pair in fold_pairs]
        plt.plot(fold_pairs, accuracies, marker='o', label=model)
    
    # Formatting the plot
    plt.title(f'Accuracy for Subject {subject}')
    plt.xlabel('Fold Pair')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)  # Assuming accuracy ranges from 0 to 100
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Model', fontsize='small')
    
    # Show the plot
    plt.tight_layout()
    plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data preparation
df = pd.DataFrame(data)  # Start with your data
df.set_index('Patient', inplace=True)

# Remove any extra spaces from column names (to avoid hidden issues)
df.columns = df.columns.str.strip()

# Melt the data into long-form
heatmap_data = df.reset_index().melt(id_vars="Patient", var_name="Model_Fold", value_name="Accuracy")

# Check for unique model names
print(heatmap_data["Model_Fold"].unique())  # Inspect the unique values

# Correctly split Model_Fold into Model and Fold
heatmap_data["Model"] = heatmap_data["Model_Fold"].str.split("_").str[:-1].str.join("_")  # Everything before the last part (i.e., 'CFC_Mu_Beta')
heatmap_data["Fold"] = heatmap_data["Model_Fold"].str.split("_").str[-1]  # Last part (i.e., '1v2', '1v3', etc.)
heatmap_data.drop("Model_Fold", axis=1, inplace=True)

# Check unique combinations of Model and Fold
print(heatmap_data[["Model", "Fold"]].drop_duplicates())  # Check for issues

# Define custom order for models
model_order = ['BPLDA', 'CSP', 'EEGNet', 'DeepConvNet']  # Preferred order for models
# Include any other models not in this list, if needed
other_models = heatmap_data['Model'].unique()
other_models = [model for model in other_models if model not in model_order]
model_order.extend(other_models)  # Append the remaining models

# Function to plot heatmap for each patient
def plot_patient_heatmap(data, **kwargs):
    pivot = data.pivot(index="Model", columns="Fold", values="Accuracy")
    sns.heatmap(
        pivot, 
        annot=True, fmt=".2f", cmap="RdBu_r", cbar=True,  # Using RdBu_r for red-blue without gray
        vmin=40, vmax=80,  # Set fixed color range between 40 and 80
        **kwargs
    )

# Faceted heatmaps by Patient
g = sns.FacetGrid(heatmap_data, col="Patient", col_wrap=4, height=4, aspect=1.2, sharex=False, sharey=False)
g.map_dataframe(plot_patient_heatmap)

# Apply model order to the heatmap
for ax in g.axes.flat:
    # Reorder the models in the plot using the custom model_order
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
    ax.set_yticklabels([label.get_text() for label in ax.get_yticklabels() if label.get_text() in model_order], rotation=0, ha="right")

# Add titles and tidy layout
g.set_titles("Patient {col_name}")
g.tight_layout()
plt.show()

#%%

data = {
        "Patient": [1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
    'BP+LDA_1v2':[45.92, 56.60, 48.61, 48.94, 63.83, 48.03, 45.49, 50.18, 49.48, 49.82, 51.36, 48.72, 49.14, 53.68, 51.06, 50.69, 45.92, 50.00, 50.90, 53.26, 50.35, 47.02, 51.04, 51.55, 58.24],
    'BP+LDA_1v3': [48.98, 51.04, 50.69, 51.42, 53.19, 49.10, 53.12, 46.32, 51.89, 49.82, 49.66, 49.82, 50.52, 49.12, 53.19, 43.40, 48.98, 51.02, 49.82, 46.74, 53.12, 49.47, 45.14, 53.49, 51.28],
    'BP+LDA_1v4': [50.34, 52.08, 52.43, 55.67, 52.13, 44.09, 48.96, 58.60, 50.17, 50.53, 
            51.36, 49.45, 51.55, 52.63, 49.29, 48.61, 53.74, 50.68, 47.31, 58.08, 
            54.51, 52.28, 51.04, 50.39, 52.01],
    'BP+LDA_1v5': [
            51.33, 51.36, 44.67, 51.18, 51.18, 51.55, 49.04, 48.48, 49.84, 49.00, 
            51.96, 52.84, 51.16, 49.16, 49.14, 48.37, 53.14, 52.61, 50.17, 48.84, 
            51.36, 49.67, 55.23, 44.80, 54.42],
    'CSP+SVM_1v2':[
        50.00, 51.04, 50.00, 54.26, 50.00, 55.91, 50.00, 51.58, 50.52, 47.37, 
        48.98, 49.45, 51.55, 50.53, 45.74, 50.00, 51.02, 51.02, 49.46, 51.55, 
        51.04, 54.74, 50.00, 50.00, 49.45
    ],
    'CSP+SVM_1v3': [
        50.00, 51.04, 50.00, 52.13, 48.94, 50.54, 51.04, 49.47, 49.48, 50.53, 
        50.00, 49.45, 49.48, 49.47, 48.94, 50.00, 50.00, 46.94, 49.46, 47.42, 
        51.04, 50.53, 50.00, 50.00, 49.45
    ],
    'CSP+SVM_1v4': [
        50.00, 51.04, 50.00, 44.68, 48.94, 50.54, 50.00, 49.47, 49.48, 50.53, 
        50.00, 49.45, 49.48, 49.47, 48.94, 50.00, 50.00, 53.06, 49.46, 49.48, 
        50.00, 50.53, 50.00, 50.00, 49.45
    ],
    'CSP+SVM_1v5': [
        51.00, 51.02, 49.00, 55.56, 50.51, 51.55, 50.00, 51.52, 49.52, 49.00, 
        51.96, 47.87, 48.51, 47.47, 48.45, 50.98, 48.51, 50.98, 48.45, 51.49, 
        51.02, 50.00, 50.00, 50.54, 50.00
    ],
    'CFC_Mu_Beta_1v2':[
        60.20, 59.38, 53.13, 52.13, 62.77, 59.14, 55.21, 58.95, 58.76, 54.74, 
        58.16, 52.75, 58.76, 57.89, 59.57, 65.63, 56.12, 61.22, 61.29, 58.76, 
        56.25, 65.26, 54.17, 60.47, 59.34
    ],
    'CFC_Mu_Beta_1v3': [
        62.24, 57.29, 61.46, 57.45, 56.38, 60.22, 62.50, 54.74, 60.82, 49.47, 
        64.29, 63.74, 60.82, 52.63, 62.77, 59.38, 63.27, 59.18, 66.67, 59.79, 
        58.33, 62.11, 58.33, 65.12, 62.64
    ],
    'CFC_Mu_Beta_1v4': [
        55.10, 60.42, 61.46, 57.45, 56.38, 51.61, 57.29, 50.53, 58.76, 58.95, 
        57.14, 63.74, 54.64, 56.84, 61.70, 54.17, 57.14, 57.14, 56.99, 57.73, 
        52.08, 58.95, 59.38, 58.14, 56.04
    ],
    'CFC_Mu_Beta_1v5': [
        61.00, 62.24, 51.00, 57.58, 53.54, 61.86, 64.42, 62.63, 60.95, 56.00, 
        57.84, 59.57, 64.36, 53.54, 50.52, 61.76, 56.44, 56.44, 56.70, 56.44, 
        57.14, 51.00, 59.80, 54.84, 52.04
    ],
    
    'MSC_1v2':[
        57.14, 70.83, 60.42, 57.45, 61.70, 50.54, 70.83, 55.79, 65.98, 56.84, 
        63.27, 59.34, 58.76, 56.84, 62.77, 61.46, 55.10, 58.16, 56.99, 57.73, 
        66.67, 56.84, 58.33, 56.98, 57.14
    ],
    'MSC_1v3': [
        58.16, 60.42, 63.54, 63.83, 58.51, 64.52, 55.21, 58.95, 54.64, 61.05, 
        55.10, 60.44, 62.89, 51.58, 63.83, 53.12, 58.16, 55.10, 56.99, 56.70, 
        60.42, 57.89, 55.21, 65.12, 60.44
    ],
    'MSC_1v4': [
        56.12, 70.83, 61.46, 55.32, 62.77, 60.22, 58.33, 55.79, 59.79, 61.05, 
        59.18, 59.34, 60.82, 54.74, 63.83, 68.75, 53.06, 65.31, 60.22, 59.79, 
        59.38, 54.74, 55.21, 53.49, 58.24
    ],
    'MSC_1v5': [
        72.00, 58.16, 61.00, 62.63, 60.61, 62.89, 59.62, 55.56, 56.19, 62.00, 
        58.82, 61.70, 65.35, 62.63, 59.79, 57.84, 60.40, 59.80, 61.86, 57.43, 
        59.18, 58.00, 63.73, 58.06, 60.20
    ],
    'PLV_1v2':[
        65.31, 69.79, 57.29, 61.70, 58.51, 56.99, 55.21, 60.00, 58.76, 70.53, 
        62.24, 61.54, 57.73, 58.95, 61.70, 63.54, 55.10, 61.22, 61.29, 61.86, 
        62.50, 55.79, 60.42, 58.14, 65.93
    ],
    'PLV_1v3': [
        64.29, 59.38, 61.46, 59.57, 59.57, 65.59, 65.62, 63.16, 53.61, 61.05, 
        59.18, 60.44, 57.73, 56.84, 58.51, 73.96, 58.16, 60.20, 60.22, 62.89, 
        59.38, 57.89, 54.17, 62.79, 65.93
    ],
    'PLV_1v4': [
        62.24, 69.79, 63.54, 56.38, 62.77, 56.99, 59.38, 70.53, 63.92, 56.84, 
        60.20, 59.34, 51.55, 61.05, 57.45, 67.71, 55.10, 70.41, 56.99, 62.89, 
        64.58, 62.11, 66.67, 58.14, 65.93
    ],
    'PLV_1v5': [
        73.00, 63.27, 57.00, 53.54, 62.63, 62.89, 62.50, 57.58, 66.67, 69.00, 
        65.69, 59.57, 56.44, 60.61, 61.86, 72.55, 56.44, 58.82, 56.70, 61.39, 
        60.20, 58.00, 64.71, 56.99, 66.33
    ],
   'CMI_1v2':[
        57.14, 77.08, 58.33, 64.89, 56.38, 54.84, 55.21, 56.84, 60.82, 60.00, 
        58.16, 58.24, 60.82, 55.79, 62.77, 63.54, 58.16, 57.14, 59.14, 71.13, 
        59.38, 60.00, 57.29, 58.14, 63.74
    ],
    'CMI_1v3': [
        62.24, 58.33, 63.54, 62.77, 64.89, 61.29, 63.54, 60.00, 55.67, 67.37, 
        58.16, 63.74, 63.92, 56.84, 59.57, 73.96, 56.12, 59.18, 56.99, 61.86, 
        61.46, 53.68, 59.38, 53.49, 69.23
    ],
    'CMI_1v4': [
        59.18, 73.96, 58.33, 62.77, 58.51, 53.76, 55.21, 64.21, 54.64, 58.95, 
        60.20, 65.93, 61.86, 53.68, 58.51, 64.58, 58.16, 63.27, 58.06, 59.79, 
        61.46, 64.21, 57.29, 58.14, 75.82
    ],
    'CMI_1v5': [
        76.00, 63.27, 68.00, 61.62, 64.65, 60.82, 61.54, 60.61, 67.62, 67.00, 
        59.80, 62.77, 67.33, 59.60, 62.89, 64.71, 59.41, 67.65, 64.95, 68.32, 
        57.14, 53.00, 71.57, 55.91, 69.39
    ],
   'EEGNet_1v2':[
        60.20, 52.08, 53.12, 60.64, 54.26, 55.91, 58.33, 56.84, 64.95, 56.84, 
        58.16, 53.85, 54.64, 55.79, 67.02, 56.25, 52.04, 57.14, 47.31, 60.82, 
        62.50, 65.26, 59.38, 61.63, 60.44
    ],
    'EEGNet_1v3': [
        57.14, 57.29, 57.29, 54.26, 53.19, 54.84, 58.33, 51.58, 56.70, 50.53, 
        60.20, 61.54, 57.73, 53.68, 55.32, 57.29, 53.06, 57.14, 54.84, 54.64, 
        53.12, 53.12, 54.17, 61.63, 62.64
    ],
    'EEGNet_1v4': [
        54.08, 56.25, 57.29, 55.32, 55.32, 56.99, 53.12, 55.79, 55.67, 49.47, 
        62.24, 54.95, 55.67, 58.95, 64.89, 55.21, 56.12, 69.39, 59.14, 58.76, 
        61.46, 54.74, 59.38, 52.33, 52.75
    ],
    'EEGNet_1v5': [
        56.00, 55.10, 54.00, 53.54, 54.55, 57.73, 55.77, 50.51, 59.05, 55.00, 
        55.88, 56.38, 58.42, 56.57, 61.86, 53.92, 59.41, 55.88, 49.48, 56.44, 
        53.06, 52.00, 55.88, 62.37, 54.08
    ],
   'DeepConvNet_1v2':[
        58.16, 60.42, 52.08, 56.38, 52.13, 58.06, 58.33, 56.84, 60.82, 52.63, 
        58.16, 58.24, 54.64, 56.84, 55.32, 57.29, 55.10, 58.16, 55.91, 49.48, 
        60.42, 55.79, 57.29, 56.98, 54.95
    ],
    'DeepConvNet_1v3': [
        54.08, 59.38, 56.25, 55.32, 60.64, 62.37, 60.42, 52.63, 58.76, 49.47, 
        55.10, 56.04, 56.70, 54.74, 58.51, 55.21, 56.12, 55.10, 59.14, 50.52, 
        53.12, 60.00, 59.38, 56.98, 65.93
    ],
    'DeepConvNet_1v4': [
        54.08, 54.17, 55.21, 51.06, 57.45, 55.91, 57.29, 57.89, 57.73, 60.00, 
        60.20, 57.14, 52.58, 51.58, 56.38, 58.33, 57.14, 59.18, 52.69, 53.61, 
        61.46, 54.74, 51.04, 54.65, 54.95
    ],
    'DeepConvNet_1v5': [
        54.00, 59.18, 61.00, 50.51, 53.54, 56.70, 55.77, 54.55, 54.29, 53.00, 
        52.94, 60.64, 57.43, 57.58, 59.79, 56.86, 52.48, 59.80, 58.76, 51.49, 
        62.24, 57.00, 53.92, 55.91, 57.14
    ]
}


# Create the dataframe
df = pd.DataFrame(data)

# Set Patient as the index
df.set_index('Patient', inplace=True)

# Create a dictionary for model titles
model_titles = {
    "BP+LDA": "Band Power with LDA",
    "CSP+SVM": "CSP with SVM",
    "CFC_Mu_Beta": "CFC (Mu and Beta)",
    "MSC": "MSC",
    "PLV": "PLV",
    "CMI": "CMI",
    "EEGNet": "EEGNet",
    "DeepConvNet": "DeepConvNet"
}

# List of models
models = ["BP+LDA", "CSP+SVM", "CFC_Mu_Beta", "MSC", "PLV", "CMI", "EEGNet", "DeepConvNet"]

# Create each figure for each model
for model in models:
    # Extract data for the current model
    model_data = df[[f"{model}_1v2", f"{model}_1v3", f"{model}_1v4", f"{model}_1v5"]]
    
    # Set up the plot
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Plot the heatmap
    sns.heatmap(model_data, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, vmin=0, vmax=100)
    
    # Formatting the plot
    plt.title(f'Accuracy Heatmap for {model_titles[model]}')
    plt.xlabel('Fold Pairs')
    plt.ylabel('Subjects')
    plt.xticks(ticks=np.arange(4) + 0.5, labels=["1v2", "1v3", "1v4", "1v5"])
    plt.yticks(ticks=np.arange(len(df.index)) + 0.5, labels=df.index, rotation=0)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
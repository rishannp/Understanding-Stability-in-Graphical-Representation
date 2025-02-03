# %% load data
import pickle
import os
import pandas as pd

# Define the directory and file name
load_dir = r"C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Understanding Stability in Graphical Representation\GAT\adjAsX\Session 1 v X Pairs\ALS\Explain\PLV"
file_name = "all_subjects_accuracies.pkl"  # Ensure this matches the saved file name
load_file_path = os.path.join(load_dir, file_name)

# Check if the file exists before attempting to load
if os.path.exists(load_file_path):
    try:
        # Load the dictionary from the pickle file
        with open(load_file_path, 'rb') as f:
            all_subjects_accuracies = pickle.load(f)
        print("Dictionary loaded successfully.")
        # Optionally print or validate the loaded data
        # print(all_subjects_accuracies)
    except Exception as e:
        print(f"Error loading dictionary: {e}")
else:
    print(f"File not found: {load_file_path}")


#%% Train Weights plotting

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define your electrode labels (22 electrodes)
electrode_labels = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7",
                    "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", "P4",
                    "P8", "O1", "O2"]

# Base directory to save the plots
base_save_path = r"C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Understanding Stability in Graphical Representation\Plots\PLV_weights\ALS"

# Iterate through each subject
for subject, train_test_pairs in all_subjects_accuracies.items():
    for pair_idx, pair_data in enumerate(train_test_pairs):
        # Extract the best weights (w3) for this train/test pair
        best_weights = pair_data['best_weights']
        w3 = best_weights['w3']  # The weights array of shape (Trials, 22)

        # Calculate number of trials
        num_trials = w3.shape[0]

        # Define the labels for trials (Left/Right)
        labels = ['Left' if i % 2 == 0 else 'Right' for i in range(num_trials)]

        # Convert 'Left'/'Right' labels to numeric values (0 for Left, 1 for Right)
        label_map = {'Left': 0, 'Right': 1}
        numeric_labels = [label_map[label] for label in labels]

        # Create directories for the subject and fold pair
        subject_save_path = os.path.join(base_save_path, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        
        fold_save_path = os.path.join(subject_save_path, f"TrainTestPair_{pair_idx}")
        os.makedirs(fold_save_path, exist_ok=True)

        # Create folder for storing the heatmaps and PCA plots
        heatmap_folder = os.path.join(fold_save_path, "Heatmaps")
        os.makedirs(heatmap_folder, exist_ok=True)

        # Plotting heatmap of the weights for each trial (electrode weights)
        plt.figure(figsize=(16, 16))  # Increase figure size for x-axis length
        sns.heatmap(
            w3,
            cmap="RdBu",
            annot=False,
            cbar=True,
            xticklabels=electrode_labels,
            yticklabels=numeric_labels,
            square=True,
            cbar_kws={'label': 'Weight Value'}  # Label for the color bar
        )
        
        # Rotate axis labels to prevent overlap
        plt.xticks(rotation=90, ha='right', fontsize=10)
        plt.yticks(ticks=range(len(numeric_labels)), labels=numeric_labels, rotation=0, fontsize=10)

        plt.title(f"Best Weights for {subject} TrainTestPair {pair_idx}", fontsize=16)
        plt.xlabel("Electrodes", fontsize=12)
        plt.ylabel("Trials (0=Left, 1=Right)", fontsize=12)
        
        # Save the heatmap plot
        heatmap_save_file = os.path.join(heatmap_folder, f"Best_Weights_Heatmap_{subject}_Pair{pair_idx}.png")
        plt.savefig(heatmap_save_file, dpi=300, bbox_inches='tight')
        plt.close()

print("All plots have been saved in their respective folders.")

#%% Understanding through weights which channels are important for each class

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define your electrode labels (22 electrodes)
electrode_labels = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7",
                    "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", "P4",
                    "P8", "O1", "O2","","",""]

# Base directory to save the plots
base_save_path = r"C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Understanding Stability in Graphical Representation\Plots\PLV_weights\ALS"

# Iterate through each subject
for subject, train_test_pairs in all_subjects_accuracies.items():
    for pair_idx, pair_data in enumerate(train_test_pairs):
        # Extract the best weights (w3) for this train/test pair
        best_weights = pair_data['best_weights']
        w3 = best_weights['w3']  # The weights array of shape (Trials, 22)

        # Debugging shapes
        print(f"w3 shape: {w3.shape}")
        print(f"electrode_labels length: {len(electrode_labels)}")
        
        # Ensure w3 dimensions match electrodes and trials
        num_trials, num_electrodes = w3.shape
        assert num_electrodes == len(electrode_labels), "Mismatch between electrode labels and w3 columns."
        
        # Define the labels for trials (Left/Right)
        labels = ['Left' if i % 2 == 0 else 'Right' for i in range(num_trials)]
        label_map = {'Left': 0, 'Right': 1}
        numeric_labels = [label_map[label] for label in labels]

        # Create directories for the subject and fold pair
        subject_save_path = os.path.join(base_save_path, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        
        fold_save_path = os.path.join(subject_save_path, f"TrainTestPair_{pair_idx}")
        os.makedirs(fold_save_path, exist_ok=True)

        # Create folder for storing the heatmaps and bar plots
        heatmap_folder = os.path.join(fold_save_path, "Heatmaps")
        barplot_folder = os.path.join(fold_save_path, "BarPlots")
        os.makedirs(heatmap_folder, exist_ok=True)
        os.makedirs(barplot_folder, exist_ok=True)

        # Plotting heatmap of the weights for each trial (electrode weights)
        plt.figure(figsize=(16, 12))  # Adjust figure size for better visibility
        sns.heatmap(
            w3,
            cmap="RdBu",
            annot=False,
            cbar=True,
            xticklabels=electrode_labels,
            yticklabels=numeric_labels,
            square=False,  # Allow flexibility in shape
            cbar_kws={'label': 'Weight Value'}  # Label for the color bar
        )
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.title(f"Best Weights for {subject} TrainTestPair {pair_idx}", fontsize=16)
        plt.xlabel("Electrodes", fontsize=12)
        plt.ylabel("Trials (0=Left, 1=Right)", fontsize=12)
        
        # Save the heatmap plot
        heatmap_save_file = os.path.join(heatmap_folder, f"Best_Weights_Heatmap_{subject}_Pair{pair_idx}.png")
        plt.savefig(heatmap_save_file, dpi=300, bbox_inches='tight')
        plt.close()

        # Calculate average weights for each class
        class_0_indices = [i for i, label in enumerate(numeric_labels) if label == 0]
        class_1_indices = [i for i, label in enumerate(numeric_labels) if label == 1]
        avg_weights_class_0 = np.mean(w3[class_0_indices], axis=0)
        avg_weights_class_1 = np.mean(w3[class_1_indices], axis=0)

        # Bar plot to visualize average weights for each electrode
        plt.figure(figsize=(14, 8))
        x = np.arange(len(electrode_labels))
        width = 0.35

        plt.bar(x - width / 2, avg_weights_class_0, width, label='Class 0 (Left)', color='blue')
        plt.bar(x + width / 2, avg_weights_class_1, width, label='Class 1 (Right)', color='red')
        plt.xticks(x, electrode_labels, rotation=45, ha='right', fontsize=10)
        plt.ylabel("Average Weight Value", fontsize=12)
        plt.xlabel("Electrodes", fontsize=12)
        plt.title(f"Average Electrode Weights for {subject} TrainTestPair {pair_idx}", fontsize=16)
        plt.legend(fontsize=12)

        # Save the bar plot
        barplot_save_file = os.path.join(barplot_folder, f"Average_Weights_BarPlot_{subject}_Pair{pair_idx}.png")
        plt.savefig(barplot_save_file, dpi=300, bbox_inches='tight')
        plt.close()

print("All plots have been saved in their respective folders.")

# Prepare data
data = []
for subject, folds in all_subjects_accuracies.items():
    for fold_data in folds:
        train_test_pair = fold_data['fold_name']
        # Using 'optimal' accuracy here, you can change it to 'mean', 'high', or 'low'
        accuracy = fold_data['optimal']
        data.append({
            'Subject': subject,  # E.g., 'S1', 'S2', etc.
            'Train-Test Pair': train_test_pair,
            'Accuracy': accuracy
        })

df = pd.DataFrame(data)


plt.figure(figsize=(14, 8))
sns.boxplot(x='Subject', y='Accuracy', data=df)
plt.title('Accuracy Distribution Across Subjects')
plt.xlabel('Subject')
plt.ylabel('Accuracy')
plt.xticks(rotation=90)  # Rotate x-axis labels if needed
plt.tight_layout()
plt.show()

# Prepare data for heatmap
heatmap_data = df.pivot(
    index='Subject', columns='Train-Test Pair', values='Accuracy')

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title('Accuracy Heatmap Across Subjects and Train-Test Pairs')
plt.xlabel('Train-Test Pair')
plt.ylabel('Subject')
plt.xticks(ticks=range(len(heatmap_data.columns)),
           labels=heatmap_data.columns, rotation=90)
plt.yticks(ticks=range(len(heatmap_data.index)),
           labels=heatmap_data.index, rotation=0)
plt.tight_layout()
plt.show()


# % Average the Accuracies

# Dictionary to hold the calculated average statistics for each subject
average_stats = {}

# Iterate over each subject in the all_subjects_accuracies dictionary
for subject, folds_list in all_subjects_accuracies.items():
    total_mean = 0
    total_high = 0
    total_low = 0
    total_optimal = 0
    num_folds = len(folds_list)

    # Iterate over each fold (dict) in the subject's list of folds
    for fold_dict in folds_list:
        total_mean += fold_dict['mean']
        total_high += fold_dict['high']
        total_low += fold_dict['low']
        total_optimal += fold_dict['optimal']

    # Calculate averages for this subject
    average_stats[subject] = {

        'Average Mean': total_mean / num_folds,
        'Average High': total_high / num_folds,
        'Average Low': total_low / num_folds,
        'Average Optimal': total_optimal / num_folds
    }

# Example: Print average statistics for each subject
for subject, averages in average_stats.items():
    print(f"Subject {subject}:")
    print(f"  Average Mean: {averages['Average Mean']:.4f}")
    print(f"  Average High: {averages['Average High']:.4f}")
    print(f"  Average Low: {averages['Average Low']:.4f}")
    print(f"  Average Optimal: {averages['Average Optimal']:.4f}")


# Initialize accumulators
total_ram_usage = 0
total_train_time = 0
total_inference_time = 0
total_folds = 0

# Iterate over all subjects and folds
for subject, folds in all_subjects_accuracies.items():
    for metrics in folds:
        total_ram_usage += metrics['ram']
        total_train_time += metrics['total_train_time']
        total_inference_time += metrics['avg_inference_time']
        total_folds += 1

# Calculate averages
avg_ram_usage = total_ram_usage / total_folds
avg_train_time = total_train_time / total_folds
avg_inference_time = total_inference_time / total_folds

# Print averages
print(f"Average RAM Usage: {avg_ram_usage:.2f} GB")
print(f"Average Training Time: {avg_train_time:.5f} seconds")
print(f"Average Inference Time: {avg_inference_time:.5f} seconds")


#%% Attention plotting
# Define the base save path for plots
save_path = r"C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Understanding Stability in Graphical Representation\GAT\adjAsX\Session 1 v X Pairs\ALS - V2\Explain"

# Electrode names for X and Y axes
electrode_labels = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7",
                    "C3", "CZ", "C4", "T8", "P7", "P3", "PZ", "P4",
                    "P8", "O1", "O2"]

# Iterate through each subject in the dictionary
for subject, train_test_pairs in all_subjects_accuracies.items():
    for pair_idx, pair_data in enumerate(train_test_pairs):
        # Extract the best attention scores (att3) for this train/test pair
        best_attention_scores = pair_data['best_attention_scores']
        att3 = best_attention_scores['att3']  # Attention scores (numpy array)

        # Calculate number of trials and reshape att3 to [22, 22, num_trials]
        num_trials = att3.shape[0] // (22 * 22)  # Calculate number of full trials

        # Reshape att3 to [22, 22, num_trials]
        reshaped_att3 = np.reshape(att3[:num_trials * 22 * 22], [22, 22, num_trials])

        # Create directories for saving plots
        layer_save_path = os.path.join(
            save_path, subject, f"TrainTestPair_{pair_idx}", "Layer_att3")
        os.makedirs(layer_save_path, exist_ok=True)

        # Loop through each trial to visualize the attention scores
        for trial_idx in range(num_trials):
            # Extract attention scores for this trial
            trial_scores = reshaped_att3[:, :, trial_idx]

            # Plot the heatmap with fixed vmin and vmax
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                trial_scores,
                cmap="RdBu",
                annot=False,
                cbar=True,
                vmin=0, vmax=1,  # Fix the color range to 0-1
                xticklabels=electrode_labels,
                yticklabels=electrode_labels,
                square=True
            )
            plt.title(
                f"{subject} TrainTestPair {pair_idx} Layer att3 Trial {trial_idx}", fontsize=14)
            plt.xlabel("Electrodes", fontsize=12)
            plt.ylabel("Electrodes", fontsize=12)

            # Save the plot
            save_file = os.path.join(
                layer_save_path, f"Trial_{trial_idx}.png")
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            plt.close()

print("All heatmaps have been plotted and saved.")

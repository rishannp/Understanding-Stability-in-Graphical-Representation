"""
Created on Tuesday 30th July 2024 at 12:22pm
Rishan Patel, Bioelectronics and Aspire Create Group, UCL


https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
"""

import os
import scipy
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.fftpack import fft, ifft

def plot_statistics(subject, means_L_C3, variances_L_C3, means_L_C4, variances_L_C4, 
                     means_R_C3, variances_R_C3, means_R_C4, variances_R_C4):
    plt.figure(figsize=(14, 12))
    
    plt.subplot(4, 1, 1)
    plt.plot(means_L_C3, label='Mean of Channel C3 (Left)')
    plt.plot(means_R_C3, label='Mean of Channel C3 (Right)')
    plt.title(f'Mean Over Trials for Channel C3 - {subject}')
    plt.xlabel('Trial')
    plt.ylabel('Mean')
    plt.legend()
    
    plt.subplot(4, 1, 2)
    plt.plot(variances_L_C3, label='Variance of Channel C3 (Left)')
    plt.plot(variances_R_C3, label='Variance of Channel C3 (Right)')
    plt.title(f'Variance Over Trials for Channel C3 - {subject}')
    plt.xlabel('Trial')
    plt.ylabel('Variance')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(means_L_C4, label='Mean of Channel C4 (Left)')
    plt.plot(means_R_C4, label='Mean of Channel C4 (Right)')
    plt.title(f'Mean Over Trials for Channel C4 - {subject}')
    plt.xlabel('Trial')
    plt.ylabel('Mean')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(variances_L_C4, label='Variance of Channel C4 (Left)')
    plt.plot(variances_R_C4, label='Variance of Channel C4 (Right)')
    plt.title(f'Variance Over Trials for Channel C4 - {subject}')
    plt.xlabel('Trial')
    plt.ylabel('Variance')
    plt.legend()

    plt.tight_layout()
    plt.show()
#%% SHU Dataset Test

# Define the directory containing your .mat files
directory = r'C:\Users\uceerjp\Desktop\PhD\SHU Dataset\MatFiles'

# Create a dictionary to store the concatenated data and labels by subject
data_by_subject = {}

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a .mat file
    if filename.endswith('.mat'):
        # Generate the full path to the file
        file_path = os.path.join(directory, filename)
        
        # Load the .mat file
        mat_data = loadmat(file_path)
        
        # Extract the subject ID and session ID from the filename
        # Example filename: sub-001_ses-01_task_motorimagery_eeg.mat
        parts = filename.split('_')
        subject_id = parts[0]  # Extract 'sub-001'
        session_id = parts[1]  # Extract 'ses-01'
        
        # Initialize the subject entry in the dictionary if not already present
        if subject_id not in data_by_subject:
            data_by_subject[subject_id] = {
                'data': [],
                'labels': []
            }
        
        # Extract data and labels from the loaded .mat file
        # Assuming 'data' and 'labels' are keys in the .mat file
        data = mat_data['data']  # Example key
        labels = mat_data['labels']  # Example key

        # Append the data and labels to the respective lists
        data_by_subject[subject_id]['data'].append(data)
        data_by_subject[subject_id]['labels'].append(labels)

# Concatenate data and labels for each subject
for subject_id in data_by_subject:
    # Concatenate the data arrays along the first dimension
    concatenated_data = np.concatenate(data_by_subject[subject_id]['data'], axis=0)
    concatenated_labels = np.concatenate(data_by_subject[subject_id]['labels'], axis=1)
    
    # Store the concatenated data and labels back in the dictionary
    data_by_subject[subject_id]['data'] = concatenated_data
    data_by_subject[subject_id]['labels'] = concatenated_labels

# Optionally: Print the concatenated data shapes for verification
for subject_id, data_info in data_by_subject.items():
    data_shape = data_info['data'].shape
    labels_shape = data_info['labels'].shape
    print(f"Subject ID: {subject_id}")
    print(f"  Data Shape: {data_shape}")
    print(f"  Labels Shape: {labels_shape}")

# Cleanup variables if necessary
del filename, file_path, mat_data, parts, subject_id, session_id, data, labels, concatenated_data, concatenated_labels, data_info, data_shape, directory, labels_shape

fs = 250

# Function to split data by label (Left and Right motor imagery)
def split_data_by_label(data_by_subject):
    data_split = {}

    for subject, data_dict in data_by_subject.items():
        # Extract data and labels
        data = data_dict['data']       # Shape: Trials x Channels x Samples
        labels = data_dict['labels']   # Shape: 1 x Trials (or Trials, depending on how it's stored)
        
        # Ensure labels are a 1D array if they are stored with shape (1, Trials)
        if labels.ndim == 2:
            labels = labels.flatten()
        
        # Initialize lists for Left and Right motor imagery
        data_L = []
        data_R = []
        
        # Iterate over trials and separate based on label
        for i in range(data.shape[0]):
            if labels[i] == 1:
                data_L.append(data[i])
            elif labels[i] == 2:
                data_R.append(data[i])
        
        # Convert lists to numpy arrays
        data_L = np.array(data_L)
        data_R = np.array(data_R)
        
        # Store split data in the dictionary
        data_split[subject] = {'L': data_L, 'R': data_R}
    
    return data_split

# Split data for Left and Right motor imagery
data_split = split_data_by_label(data_by_subject)

# Function to compute statistics (mean and variance) for channels C3 and C4
def compute_statistics(data):
    means_C3 = np.mean(data[:, 11, :], axis=1)  # Mean across trials for channel C3
    variances_C3 = np.var(data[:, 11, :], axis=1)  # Variance across trials for channel C3
    
    means_C4 = np.mean(data[:, 12, :], axis=1)  # Mean across trials for channel C4
    variances_C4 = np.var(data[:, 12, :], axis=1)  # Variance across trials for channel C4
    
    return means_C3, variances_C3, means_C4, variances_C4

# Function to plot statistics with automatic y-axis bounds adjustment
def plot_statistics(subject, means_L_C3, variances_L_C3, means_L_C4, variances_L_C4, 
                    means_R_C3, variances_R_C3, means_R_C4, variances_R_C4):
    # Find the minimum number of trials between Left and Right to avoid dimension mismatch
    min_trials = min(len(means_L_C3), len(means_R_C3))
    
    # Truncate the data for both Left and Right to the minimum number of trials
    means_L_C3 = means_L_C3[:min_trials]
    variances_L_C3 = variances_L_C3[:min_trials]
    means_L_C4 = means_L_C4[:min_trials]
    variances_L_C4 = variances_L_C4[:min_trials]
    
    means_R_C3 = means_R_C3[:min_trials]
    variances_R_C3 = variances_R_C3[:min_trials]
    means_R_C4 = means_R_C4[:min_trials]
    variances_R_C4 = variances_R_C4[:min_trials]
    
    trials = np.arange(min_trials)  # Use the minimum trial count for the x-axis

    plt.figure(figsize=(12, 8))

    # Plot Channel C3 with dual y-axes (Left & Right)
    plt.subplot(2, 1, 1)
    ax1 = plt.gca()  # Primary axis for means
    ax2 = ax1.twinx()  # Secondary axis for variances

    ax1.plot(trials, means_L_C3, label='Mean of C3 (Left)', color='blue', linestyle='-')
    ax1.plot(trials, means_R_C3, label='Mean of C3 (Right)', color='orange', linestyle='-')
    ax2.fill_between(trials, variances_L_C3, color='blue', alpha=0.2, label='Variance of C3 (Left)')
    ax2.fill_between(trials, variances_R_C3, color='orange', alpha=0.2, label='Variance of C3 (Right)')
    
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Mean', color='black')
    ax2.set_ylabel('Variance', color='black')
    ax1.set_title(f'Channel C3 - {subject}')
    
    # Automatically adjust the y-axis limits based on the data
    ax1.set_ylim([min(min(means_L_C3), min(means_R_C3)) - 1e-7, 
                  max(max(means_L_C3), max(means_R_C3)) + 1e-7])
    ax2.set_ylim([min(min(variances_L_C3), min(variances_R_C3)) - 10, 
                  max(max(variances_L_C3), max(variances_R_C3)) + 10])
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Plot Channel C4 with dual y-axes (Left & Right)
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()  # Primary axis for means
    ax2 = ax1.twinx()  # Secondary axis for variances

    ax1.plot(trials, means_L_C4, label='Mean of C4 (Left)', color='blue', linestyle='-')
    ax1.plot(trials, means_R_C4, label='Mean of C4 (Right)', color='orange', linestyle='-')
    ax2.fill_between(trials, variances_L_C4, color='blue', alpha=0.2, label='Variance of C4 (Left)')
    ax2.fill_between(trials, variances_R_C4, color='orange', alpha=0.2, label='Variance of C4 (Right)')
    
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Mean', color='black')
    ax2.set_ylabel('Variance', color='black')
    ax1.set_title(f'Channel C4 - {subject}')
    
    # Automatically adjust the y-axis limits based on the data
    ax1.set_ylim([min(min(means_L_C4), min(means_R_C4)) - 1e-7, 
                  max(max(means_L_C4), max(means_R_C4)) + 1e-7])
    ax2.set_ylim([min(min(variances_L_C4), min(variances_R_C4)) - 10, 
                  max(max(variances_L_C4), max(variances_R_C4)) + 10])
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()





# Plot the data for each subject
for subject, data_dict in data_split.items():
    data_L = data_dict['L']  # Data for Left Motor Imagery
    data_R = data_dict['R']  # Data for Right Motor Imagery

    # Compute statistics for channels C3 and C4
    means_L_C3, variances_L_C3, means_L_C4, variances_L_C4 = compute_statistics(data_L)
    means_R_C3, variances_R_C3, means_R_C4, variances_R_C4 = compute_statistics(data_R)

    # Plot statistics
    plot_statistics(subject, means_L_C3, variances_L_C3, means_L_C4, variances_L_C4, 
                     means_R_C3, variances_R_C3, means_R_C4, variances_R_C4)


#%% SHU Dataset
channels = np.array([
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5',
    'FC6', 'CZ', 'C3', 'C4', 'T3', 'T4', 'A1', 'A2', 'CP1', 'CP2',
    'CP5', 'CP6', 'PZ', 'P3', 'P4', 'T5', 'T6', 'PO3', 'PO4', 'OZ',
    'O1', 'O2'
])

# Define the directory containing your .mat files
directory = r'C:\Users\uceerjp\Desktop\PhD\SHU Dataset\MatFiles'

# Create a dictionary to store the concatenated data and labels by subject
data_by_subject = {}

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a .mat file
    if filename.endswith('.mat'):
        # Generate the full path to the file
        file_path = os.path.join(directory, filename)
        
        # Load the .mat file
        mat_data = loadmat(file_path)
        
        # Extract the subject ID and session ID from the filename
        # Example filename: sub-001_ses-01_task_motorimagery_eeg.mat
        parts = filename.split('_')
        subject_id = parts[0]  # Extract 'sub-001'
        session_id = parts[1]  # Extract 'ses-01'
        
        # Initialize the subject entry in the dictionary if not already present
        if subject_id not in data_by_subject:
            data_by_subject[subject_id] = {
                'data': [],
                'labels': []
            }
        
        # Extract data and labels from the loaded .mat file
        # Assuming 'data' and 'labels' are keys in the .mat file
        data = mat_data['data']  # Example key
        labels = mat_data['labels']  # Example key

        # Append the data and labels to the respective lists
        data_by_subject[subject_id]['data'].append(data)
        data_by_subject[subject_id]['labels'].append(labels)

# Concatenate data and labels for each subject
for subject_id in data_by_subject:
    # Concatenate the data arrays along the first dimension
    concatenated_data = np.concatenate(data_by_subject[subject_id]['data'], axis=0)
    concatenated_labels = np.concatenate(data_by_subject[subject_id]['labels'], axis=1)
    
    # Store the concatenated data and labels back in the dictionary
    data_by_subject[subject_id]['data'] = concatenated_data
    data_by_subject[subject_id]['labels'] = concatenated_labels

# Optionally: Print the concatenated data shapes for verification
for subject_id, data_info in data_by_subject.items():
    data_shape = data_info['data'].shape
    labels_shape = data_info['labels'].shape
    print(f"Subject ID: {subject_id}")
    print(f"  Data Shape: {data_shape}")
    print(f"  Labels Shape: {labels_shape}")

# Cleanup variables if necessary
del filename, file_path, mat_data, parts, subject_id, session_id, data, labels, concatenated_data, concatenated_labels, data_info, data_shape, directory, labels_shape

fs = 250

# Split Left and Rights

def split_data_by_label(data_by_subject):
    data_split = {}

    for subject, data_dict in data_by_subject.items():
        # Extract data and labels
        data = data_dict['data']       # Shape: Trials x Channels x Samples
        labels = data_dict['labels']   # Shape: 1 x Trials (or Trials, depending on how it's stored)
        
        # Ensure labels are a 1D array if they are stored with shape (1, Trials)
        if labels.ndim == 2:
            labels = labels.flatten()
        
        # Initialize lists for Left and Right motor imagery
        data_L = []
        data_R = []
        
        # Iterate over trials and separate based on label
        for i in range(data.shape[0]):
            if labels[i] == 1:
                data_L.append(data[i])
            elif labels[i] == 2:
                data_R.append(data[i])
        
        # Convert lists to numpy arrays
        data_L = np.array(data_L)
        data_R = np.array(data_R)
        
        # Store split data in the dictionary
        data_split[subject] = {'L': data_L, 'R': data_R}
    
    return data_split

# Example usage:
# data_by_subject = load_data()  # Load your data dictionary here
data_split = split_data_by_label(data_by_subject)

# `data_split` will contain the split data for each subject
# Each subject's entry will have 'data_L' and 'data_R' for Left and Right motor imagery

#% Mean, Variance and Auto-correlation plot over all trial for each subjects


def compute_statistics(data):
    # Compute mean and variance for channels C3 and C4
    means_C3 = np.mean(data[:, 11, :], axis=1)  # Mean across trials for channel C3
    variances_C3 = np.var(data[:, 11, :], axis=1)  # Variance across trials for channel C3
    
    means_C4 = np.mean(data[:, 12, :], axis=1)  # Mean across trials for channel C4
    variances_C4 = np.var(data[:, 12, :], axis=1)  # Variance across trials for channel C4
    
    return means_C3, variances_C3, means_C4, variances_C4


def test_stationarity(signal):
    adf_result = adfuller(signal)
    kpss_result = kpss(signal, regression='c')
    return adf_result, kpss_result

# Dictionaries to store results
results = {
    'subject': [],
    'left_adf_C3': [],
    'left_adf_pvalue_C3': [],
    'left_kpss_C3': [],
    'left_kpss_pvalue_C3': [],
    'right_adf_C3': [],
    'right_adf_pvalue_C3': [],
    'right_kpss_C3': [],
    'right_kpss_pvalue_C3': [],
    'left_adf_C4': [],
    'left_adf_pvalue_C4': [],
    'left_kpss_C4': [],
    'left_kpss_pvalue_C4': [],
    'right_adf_C4': [],
    'right_adf_pvalue_C4': [],
    'right_kpss_C4': [],
    'right_kpss_pvalue_C4': []
}

# Example usage:
# Assuming `data_split` is your split data dictionary
for subject, data_dict in data_split.items():
    data_L = data_dict['L']  # Data for Left Motor Imagery
    data_R = data_dict['R']  # Data for Right Motor Imagery

    # Compute statistics for channels C3 and C4
    means_L_C3, variances_L_C3, means_L_C4, variances_L_C4 = compute_statistics(data_L)
    means_R_C3, variances_R_C3, means_R_C4, variances_R_C4 = compute_statistics(data_R)

    # Plot statistics
    plot_statistics(subject, means_L_C3, variances_L_C3, means_L_C4, variances_L_C4, 
                     means_R_C3, variances_R_C3, means_R_C4, variances_R_C4)

    # Perform stationarity tests on a sample trial for channels C3 and C4
    sample_signal_L_C3 = data_L[0, 11, :]  # Example: first trial, channel C3 for Left Motor Imagery
    sample_signal_R_C3 = data_R[0, 11, :]  # Example: first trial, channel C3 for Right Motor Imagery
    sample_signal_L_C4 = data_L[0, 12, :]  # Example: first trial, channel C4 for Left Motor Imagery
    sample_signal_R_C4 = data_R[0, 12, :]  # Example: first trial, channel C4 for Right Motor Imagery
    
    adf_result_L_C3, kpss_result_L_C3 = test_stationarity(sample_signal_L_C3)
    adf_result_R_C3, kpss_result_R_C3 = test_stationarity(sample_signal_R_C3)
    adf_result_L_C4, kpss_result_L_C4 = test_stationarity(sample_signal_L_C4)
    adf_result_R_C4, kpss_result_R_C4 = test_stationarity(sample_signal_R_C4)
    
    # Store results
    results['subject'].append(subject)
    results['left_adf_C3'].append(adf_result_L_C3[0])
    results['left_adf_pvalue_C3'].append(adf_result_L_C3[1])
    results['left_kpss_C3'].append(kpss_result_L_C3[0])
    results['left_kpss_pvalue_C3'].append(kpss_result_L_C3[1])
    results['right_adf_C3'].append(adf_result_R_C3[0])
    results['right_adf_pvalue_C3'].append(adf_result_R_C3[1])
    results['right_kpss_C3'].append(kpss_result_R_C3[0])
    results['right_kpss_pvalue_C3'].append(kpss_result_R_C3[1])
    results['left_adf_C4'].append(adf_result_L_C4[0])
    results['left_adf_pvalue_C4'].append(adf_result_L_C4[1])
    results['left_kpss_C4'].append(kpss_result_L_C4[0])
    results['left_kpss_pvalue_C4'].append(kpss_result_L_C4[1])
    results['right_adf_C4'].append(adf_result_R_C4[0])
    results['right_adf_pvalue_C4'].append(adf_result_R_C4[1])
    results['right_kpss_C4'].append(kpss_result_R_C4[0])
    results['right_kpss_pvalue_C4'].append(kpss_result_R_C4[1])

# Print stored results
for idx, subject in enumerate(results['subject']):
    print(f"Subject: {subject}")
    print(f"  Left Motor Imagery - Channel C3 - ADF Statistic: {results['left_adf_C3'][idx]}, p-value: {results['left_adf_pvalue_C3'][idx]}")
    print(f"  Left Motor Imagery - Channel C3 - KPSS Statistic: {results['left_kpss_C3'][idx]}, p-value: {results['left_kpss_pvalue_C3'][idx]}")
    print(f"  Right Motor Imagery - Channel C3 - ADF Statistic: {results['right_adf_C3'][idx]}, p-value: {results['right_adf_pvalue_C3'][idx]}")
    print(f"  Right Motor Imagery - Channel C3 - KPSS Statistic: {results['right_kpss_C3'][idx]}, p-value: {results['right_kpss_pvalue_C3'][idx]}")
    print(f"  Left Motor Imagery - Channel C4 - ADF Statistic: {results['left_adf_C4'][idx]}, p-value: {results['left_adf_pvalue_C4'][idx]}")
    print(f"  Left Motor Imagery - Channel C4 - KPSS Statistic: {results['left_kpss_C4'][idx]}, p-value: {results['left_kpss_pvalue_C4'][idx]}")
    print(f"  Right Motor Imagery - Channel C4 - ADF Statistic: {results['right_adf_C4'][idx]}, p-value: {results['right_adf_pvalue_C4'][idx]}")
    print(f"  Right Motor Imagery - Channel C4 - KPSS Statistic: {results['right_kpss_C4'][idx]}, p-value: {results['right_kpss_pvalue_C4'][idx]}")


#%% ALS Work
import os
import scipy
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.fftpack import fft, ifft

def load_and_aggregate_data(directory, file_prefix, num_files):
    aggregated_data = {}
    
    # Iterate over the file indices
    for i in range(1, num_files + 1):
        file_name = f'{file_prefix}{i}.mat'
        file_path = os.path.join(directory, file_name)
        
        # Check if the file exists
        if os.path.isfile(file_path):
            print(f'Loading {file_path}')
            mat_data = scipy.io.loadmat(file_path)
            
            # Extract data for each subject
            for key in mat_data.keys():
                if key.startswith('Subject'):
                    subject_key = key
                    if subject_key not in aggregated_data:
                        aggregated_data[subject_key] = mat_data[key]
                    else:
                        # If you want to handle cases where the same subject might appear in multiple files
                        # (e.g., concatenating arrays), you can add code here.
                        # For now, we'll just overwrite with the latest data.
                        aggregated_data[subject_key] = mat_data[key]
        else:
            print(f'{file_path} does not exist.')
    
    return aggregated_data

# Directory where .mat files are located
directory = r'C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Data'

# Prefix and number of files
file_prefix = 'OGFS'
num_files = 39

# Load and aggregate the data
data = load_and_aggregate_data(directory, file_prefix, num_files)

# Example: print the keys of the aggregated data
print(f"Aggregated data keys: {list(data.keys())}")

del directory, file_prefix, num_files
 
channel = np.array(['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2'])
#%%

# def plot_statistics(subject, means_L_C3, variances_L_C3, means_R_C3, variances_R_C3,
#                     means_L_C4, variances_L_C4, means_R_C4, variances_R_C4, mean_scale_factor=1e6):
#     """
#     Plot means with variance on a second y-axis in a 2x2 grid, scaling the means appropriately.
#     """

#     # Create separate trial arrays for left and right (to handle different lengths)
#     trials_L_C3 = np.arange(len(means_L_C3))
#     trials_R_C3 = np.arange(len(means_R_C3))
#     trials_L_C4 = np.arange(len(means_L_C4))
#     trials_R_C4 = np.arange(len(means_R_C4))

#     fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#     fig.suptitle(f'Subject {subject}', fontsize=16)
    
#     # C3 (Left)
#     ax1 = axes[0, 0]
#     ax1.plot(trials_L_C3, means_L_C3 * mean_scale_factor, label='Mean of C3 (Left)', color='blue')  # Scaling the mean
#     ax1.fill_between(trials_L_C3, (means_L_C3 - variances_L_C3),
#                      (means_L_C3 + variances_L_C3), color='blue', alpha=0.2)
#     ax1.set_xlabel('Trial')
#     ax1.set_ylabel(f'Mean (scaled by {mean_scale_factor})')
#     ax1.grid(True)

#     # Secondary y-axis for variance
#     ax2 = ax1.twinx()
#     ax2.plot(trials_L_C3, variances_L_C3, color='red', linestyle='', label='Variance (Left)')
#     ax2.set_ylabel('Variance')

#     # C3 (Right)
#     ax1 = axes[0, 1]
#     ax1.plot(trials_R_C3, means_R_C3 * mean_scale_factor, label='Mean of C3 (Right)', color='orange')  # Scaling the mean
#     ax1.fill_between(trials_R_C3, (means_R_C3 - variances_R_C3),
#                      (means_R_C3 + variances_R_C3), color='orange', alpha=0.2)
#     ax1.set_xlabel('Trial')
#     ax1.set_ylabel(f'Mean (scaled by {mean_scale_factor})')
#     ax1.grid(True)

#     # Secondary y-axis for variance
#     ax2 = ax1.twinx()
#     ax2.plot(trials_R_C3, variances_R_C3, color='red', linestyle='', label='Variance (Right)')
#     ax2.set_ylabel('Variance')

#     # C4 (Left)
#     ax1 = axes[1, 0]
#     ax1.plot(trials_L_C4, means_L_C4 * mean_scale_factor, label='Mean of C4 (Left)', color='green')  # Scaling the mean
#     ax1.fill_between(trials_L_C4, (means_L_C4 - variances_L_C4),
#                      (means_L_C4 + variances_L_C4), color='green', alpha=0.2)
#     ax1.set_xlabel('Trial')
#     ax1.set_ylabel(f'Mean (scaled by {mean_scale_factor})')
#     ax1.grid(True)

#     # Secondary y-axis for variance
#     ax2 = ax1.twinx()
#     ax2.plot(trials_L_C4, variances_L_C4, color='red', linestyle='', label='Variance (Left)')
#     ax2.set_ylabel('Variance')

#     # C4 (Right)
#     ax1 = axes[1, 1]
#     ax1.plot(trials_R_C4, means_R_C4 * mean_scale_factor, label='Mean of C4 (Right)', color='purple')  # Scaling the mean
#     ax1.fill_between(trials_R_C4, (means_R_C4 - variances_R_C4),
#                      (means_R_C4 + variances_R_C4), color='purple', alpha=0.2)
#     ax1.set_xlabel('Trial')
#     ax1.set_ylabel(f'Mean (scaled by {mean_scale_factor})')
#     ax1.grid(True)

#     # Secondary y-axis for variance
#     ax2 = ax1.twinx()
#     ax2.plot(trials_R_C4, variances_R_C4, color='red', linestyle='', label='Variance (Right)')
#     ax2.set_ylabel('Variance')

#     plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for title space
#     plt.show()

# Function to plot statistics with automatic y-axis bounds adjustment
def plot_statistics(subject, means_L_C3, variances_L_C3, means_L_C4, variances_L_C4, 
                    means_R_C3, variances_R_C3, means_R_C4, variances_R_C4):
    # Find the minimum number of trials between Left and Right to avoid dimension mismatch
    min_trials = min(len(means_L_C3), len(means_R_C3))
    
    # Truncate the data for both Left and Right to the minimum number of trials
    means_L_C3 = means_L_C3[:min_trials]
    variances_L_C3 = variances_L_C3[:min_trials]
    means_L_C4 = means_L_C4[:min_trials]
    variances_L_C4 = variances_L_C4[:min_trials]
    
    means_R_C3 = means_R_C3[:min_trials]
    variances_R_C3 = variances_R_C3[:min_trials]
    means_R_C4 = means_R_C4[:min_trials]
    variances_R_C4 = variances_R_C4[:min_trials]
    
    trials = np.arange(min_trials)  # Use the minimum trial count for the x-axis

    plt.figure(figsize=(12, 8))

    # Plot Channel C3 with dual y-axes (Left & Right)
    plt.subplot(2, 1, 1)
    ax1 = plt.gca()  # Primary axis for means
    ax2 = ax1.twinx()  # Secondary axis for variances

    ax1.plot(trials, means_L_C3, label='Mean of C3 (Left)', color='blue', linestyle='-')
    ax1.plot(trials, means_R_C3, label='Mean of C3 (Right)', color='orange', linestyle='-')
    ax2.fill_between(trials, variances_L_C3, color='blue', alpha=0.2, label='Variance of C3 (Left)')
    ax2.fill_between(trials, variances_R_C3, color='orange', alpha=0.2, label='Variance of C3 (Right)')
    
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Mean', color='black')
    ax2.set_ylabel('Variance', color='black')
    ax1.set_title(f'Channel C3 - {subject}')
    
    # Automatically adjust the y-axis limits based on the data
    ax1.set_ylim([min(min(means_L_C3), min(means_R_C3)) - 1e-7, 
                  max(max(means_L_C3), max(means_R_C3)) + 1e-7])
    ax2.set_ylim([min(min(variances_L_C3), min(variances_R_C3)) - 10, 
                  max(max(variances_L_C3), max(variances_R_C3)) + 10])
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Plot Channel C4 with dual y-axes (Left & Right)
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()  # Primary axis for means
    ax2 = ax1.twinx()  # Secondary axis for variances

    ax1.plot(trials, means_L_C4, label='Mean of C4 (Left)', color='blue', linestyle='-')
    ax1.plot(trials, means_R_C4, label='Mean of C4 (Right)', color='orange', linestyle='-')
    ax2.fill_between(trials, variances_L_C4, color='blue', alpha=0.2, label='Variance of C4 (Left)')
    ax2.fill_between(trials, variances_R_C4, color='orange', alpha=0.2, label='Variance of C4 (Right)')
    
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Mean', color='black')
    ax2.set_ylabel('Variance', color='black')
    ax1.set_title(f'Channel C4 - {subject}')
    
    # Automatically adjust the y-axis limits based on the data
    ax1.set_ylim([min(min(means_L_C4), min(means_R_C4)) - 1e-7, 
                  max(max(means_L_C4), max(means_R_C4)) + 1e-7])
    ax2.set_ylim([min(min(variances_L_C4), min(variances_R_C4)) - 10, 
                  max(max(variances_L_C4), max(variances_R_C4)) + 10])
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

for key in data:
    subject_data = data[key]
    data_L = subject_data['L']  # List of trials for Left Motor Imagery
    data_R = subject_data['R']  # List of trials for Right Motor Imagery

    num_trials_L = data_L.shape[1]
    num_trials_R = data_R.shape[1]
    
    # Initialize arrays to hold means and variances for the Left trials
    mean_lC3 = np.zeros(num_trials_L)
    mean_lC4 = np.zeros(num_trials_L)
    var_lC3 = np.zeros(num_trials_L)
    var_lC4 = np.zeros(num_trials_L)
    
    # Initialize arrays to hold means and variances for the Right trials
    mean_rC3 = np.zeros(num_trials_R)
    mean_rC4 = np.zeros(num_trials_R)
    var_rC3 = np.zeros(num_trials_R)
    var_rC4 = np.zeros(num_trials_R)
    
    # Loop through the Left trials and calculate mean and variance for C3 and C4
    for i in range(num_trials_L):
        l = data_L[0, i]  # Shape: (samples, channels)
        mean_lC3[i] = np.mean(l[:, 8])  # C3
        mean_lC4[i] = np.mean(l[:, 10])  # C4
        var_lC3[i] = np.var(l[:, 8])  # Variance of C3
        var_lC4[i] = np.var(l[:, 10])  # Variance of C4

    # Loop through the Right trials and calculate mean and variance for C3 and C4
    for i in range(num_trials_R):
        r = data_R[0, i]  # Shape: (samples, channels)
        mean_rC3[i] = np.mean(r[:, 8])  # C3
        mean_rC4[i] = np.mean(r[:, 10])  # C4
        var_rC3[i] = np.var(r[:, 8])  # Variance of C3
        var_rC4[i] = np.var(r[:, 10])  # Variance of C4
    
    # Call the plot_statistics function to plot the means and variances
    plot_statistics(
        key,
        mean_lC3, var_lC3,  # Left C3
        mean_rC3, var_rC3,  # Right C3
        mean_lC4, var_lC4,  # Left C4
        mean_rC4, var_rC4   # Right C4
    )


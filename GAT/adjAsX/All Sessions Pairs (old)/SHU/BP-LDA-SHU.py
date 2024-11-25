# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:26:26 2024
Rishan Patel, UCL, Bioelectronics Group.



https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
"""
import os
from os.path import dirname, join as pjoin
import scipy as sp
import scipy.io as sio
from scipy import signal
import numpy as np
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import networkx as nx
import torch as torch
from scipy.signal import welch, coherence, hilbert, butter, filtfilt
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.integrate import simps
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GAT, GraphNorm
from torch_geometric.nn import global_mean_pool
from torch import nn
import pickle
from progressbar import progressbar
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

#%% Functions
def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = sig.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sig.sosfiltfilt(sos, data)
    return filtered_data

def bandpower(data,low,high):

    fs = 250
    # Define window length (2s)
    win = 2* fs
    freqs, psd = signal.welch(data, fs, nperseg=win)
    
    # Find intersecting values in frequency vector
    idx_delta = np.logical_and(freqs >= low, freqs <= high)
    
    # # Plot the power spectral density and fill the delta area
    # plt.figure(figsize=(7, 4))
    # plt.plot(freqs, psd, lw=2, color='k')
    # plt.fill_between(freqs, psd, where=idx_delta, color='skyblue')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power spectral density (uV^2 / Hz)')
    # plt.xlim([0, 40])
    # plt.ylim([0, psd.max() * 1.1])
    # plt.title("Welch's periodogram")
    
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
    
    # Compute the absolute power by approximating the area under the curve
    power = simps(psd[idx_delta], dx=freq_res)
    
    return power

def bandpowercalc(l,band,fs):   
    x = np.zeros([l.shape[0],l.shape[3],l.shape[2]])
    for i in range(l.shape[0]): #node
        for j in range(l.shape[2]): #sample
            for k in range(0,l.shape[3]): #band
                data = l[i,:,j,k]
                low = band[k]
                high = band[k+1]
                x[i,k,j] = bandpower(data,low,high)

    return x

#%%

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


def bandpass_filter_trials(data_split, low_freq, high_freq, sfreq):
    filtered_data_split = {}

    # Design the bandpass filter
    nyquist = 0.5 * sfreq
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='band')

    for subject in data_split:
        subject_data = data_split[subject]
        filtered_subject_data = {}

        for direction in ['L', 'R']:
            trials = subject_data[direction]  # Shape: (trials, channels, samples)
            filtered_trials = []

            for trial in range(trials.shape[0]):
                trial_data = trials[trial, :, :]  # Shape: (channels, samples)

                # Apply bandpass filter to each channel
                filtered_trial_data = np.zeros_like(trial_data)
                for ch in range(trial_data.shape[0]):
                    filtered_trial_data[ch, :] = filtfilt(b, a, trial_data[ch, :])
                
                filtered_trials.append(filtered_trial_data)

            # Convert the list of filtered trials back to a NumPy array
            filtered_subject_data[direction] = np.array(filtered_trials)

        filtered_data_split[subject] = filtered_subject_data

    return filtered_data_split

# Apply bandpass filter to the data_split
filtered_data_split = bandpass_filter_trials(data_split, low_freq=8, high_freq=30, sfreq=fs)

# Optionally delete the original data to free up memory
del data_split, data_by_subject


def split_into_folds(data, n_splits=5):
    """Splits data into n_splits causal folds."""
    n_trials = data.shape[0]
    trials_per_fold = n_trials // n_splits
    
    # Initialize fold lists
    folds = [None] * n_splits
    
    for i in range(n_splits):
        start_index = i * trials_per_fold
        if i == n_splits - 1:
            end_index = n_trials  # Last fold gets all remaining trials
        else:
            end_index = (i + 1) * trials_per_fold
        
        folds[i] = data[start_index:end_index]
    
    return folds

def prepare_folds_for_subjects(filtered_data_split, n_splits=5):
    """Splits filtered data into causal folds for each subject and direction."""
    fold_data = {}

    for subject in filtered_data_split:
        subject_data = filtered_data_split[subject]
        fold_data[subject] = {}
        
        for direction in ['L', 'R']:
            trials = subject_data[direction]  # Shape: (trials, channels, samples)
            folds = split_into_folds(trials, n_splits=n_splits)
            
            fold_data[subject][direction] = folds
    
    return fold_data

# Apply the fold splitting to filtered data
fold_data = prepare_folds_for_subjects(filtered_data_split)

# Example of accessing the folds
for subject, data in fold_data.items():
    for direction, folds in data.items():
        for i, fold in enumerate(folds):
            print(f"Subject: {subject}, Direction: {direction}, Fold {i+1} shape: {fold.shape}")

# Initialize the new dictionary to store the merged data
merged_fold_data = {}

# Iterate over each subject in the original fold_data
for subject_id, sessions in fold_data.items():
    # Initialize each subject's dictionary
    merged_fold_data[subject_id] = {}
    
    # Iterate over each fold within "L" and "R"
    for fold_id in range(5):  # Assuming there are exactly 5 folds per session
        # Retrieve the data for the current fold in "L" and "R"
        left_data = sessions['L'][fold_id]  # Shape: Trials x Channels x Samples
        right_data = sessions['R'][fold_id]  # Shape: Trials x Channels x Samples
        
        # Concatenate the data along the Trials dimension
        combined_data = np.concatenate((left_data, right_data), axis=0)
        
        # Create labels: 0 for "L" trials and 1 for "R" trials
        left_labels = np.zeros(left_data.shape[0], dtype=int)  # Label 0 for each L trial
        right_labels = np.ones(right_data.shape[0], dtype=int)  # Label 1 for each R trial
        
        # Concatenate labels
        combined_labels = np.concatenate((left_labels, right_labels), axis=0)
        
        # Store the combined data and labels in the new structure
        merged_fold_data[subject_id][fold_id] = {
            'data': combined_data,
            'label': combined_labels
        }

# merged_fold_data now contains the merged structure with data and labels

del combined_data, combined_labels, data, direction, filtered_data_split, fold, fold_data, fold_id, folds, i, left_data, left_labels, right_data, right_labels, sessions, subject, subject_id
#%%

import numpy as np

# Bandpass filter function
def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    # Apply the filter along the sample axis (-1)
    filtered_data = signal.sosfiltfilt(sos, data, axis=-1)
    return filtered_data

# Compute band power for a single trial
def bandpower(data: np.ndarray, low: float, high: float, fs: float):
    # Define window length (2 seconds)
    win = 2 * fs
    # Compute the power spectral density (PSD)
    freqs, psd = signal.welch(data, fs, nperseg=win)
    # Find indices corresponding to the frequency range
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    # Calculate power as the area under the curve in the selected range
    freq_res = freqs[1] - freqs[0]
    power = simps(psd[idx_band], dx=freq_res)
    return power

def compute_band_power_fixed_window(merged_fold_data, bands, sample_rate=250, step_size=1):
    """
    Compute band power with a fixed 2-second window length and a sliding window.

    Args:
    - merged_fold_data: Original data dictionary.
    - bands: List of band edges, e.g., [8, 12, 16, ...].
    - sample_rate: Sampling rate of the data in Hz.
    - step_size: Step size for sliding window in seconds (default: 1 second).

    Returns:
    - output_data: Dictionary with band power, retaining the time dimension.
    """
    output_data = {}

    # Fixed window length: 2 seconds
    win_samples = 2 * sample_rate
    step_samples = int(step_size * sample_rate)

    for subject, sessions in merged_fold_data.items():
        output_data[subject] = {}
        
        for session_id, session_data in sessions.items():
            data = session_data['data']  # Trials x Channels x Samples
            labels = session_data['label']

            num_trials, num_channels, num_samples = data.shape
            num_bands = len(bands) - 1

            # Compute number of windows per trial
            num_windows = (num_samples - win_samples) // step_samples + 1

            # Initialize output array: Trials x Channels x Time Windows x Bands
            band_data = np.zeros((num_trials, num_channels, num_windows, num_bands))

            # Iterate over trials
            for trial_idx in range(num_trials):
                for channel_idx in range(num_channels):
                    trial_data = data[trial_idx, channel_idx, :]

                    # Sliding window
                    for win_idx in range(num_windows):
                        start = win_idx * step_samples
                        end = start + win_samples
                        segment = trial_data[start:end]

                        # Compute band power for this segment
                        for band_idx in range(num_bands):
                            low, high = bands[band_idx], bands[band_idx + 1]
                            band_data[trial_idx, channel_idx, win_idx, band_idx] = bandpower(segment, low, high, sample_rate)
            
            # Save computed band power and labels
            output_data[subject][session_id] = {
                'data': band_data,  # Trials x Channels x Time Windows x Bands
                'label': labels
            }

    return output_data


bands = list(range(8, 41, 4))  # Frequency bands: [8-12, 12-16, ..., 36-40]
sample_rate = 250  # Sampling rate of the EEG data
step_size = 1      # 50% overlap with 2-second window

band_power_data = compute_band_power_fixed_window(
    merged_fold_data,
    bands,
    sample_rate=sample_rate,
    step_size=step_size
)


#%%

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def plot_tsne_band_power(band_power_data):
    """
    Generate t-SNE plots for each session in the band power data.

    Args:
    - band_power_data: Dictionary with structure:
      Subject -> Session -> {'data': Trials x Channels x Time Windows x Bands, 'label': Labels}
    """
    # Iterate over each subject
    for subject, sessions in band_power_data.items():
        num_sessions = len(sessions)
        fig, axes = plt.subplots(1, num_sessions, figsize=(24, 8), sharex=True, sharey=True)
        
        # Iterate over each session for the subject
        for session_idx, (session_id, session_data) in enumerate(sessions.items()):
            # Extract band power data and labels
            data = session_data['data']  # Shape: Trials x Channels x Time Windows x Bands
            labels = session_data['label']  # Corresponding labels
            
            # Reshape data for t-SNE: (Trials x Channels x Time Windows x Bands) -> (num_samples, num_features)
            num_trials, num_channels, num_windows, num_bands = data.shape
            reshaped_data = data.transpose(0, 2, 1, 3).reshape(num_trials * num_windows, num_channels * num_bands)
            
            # Repeat labels for time windows to match reshaped data
            repeated_labels = np.repeat(labels, num_windows)
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=0)
            tsne_results = tsne.fit_transform(reshaped_data)
            
            # Plot t-SNE results
            ax = axes[session_idx]
            scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=repeated_labels, cmap='viridis', s=20)
            ax.set_title(f'Session {session_id}', fontsize=16)
            ax.axis('off')
        
        # Add a colorbar to the last subplot
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cbar = plt.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('Labels', fontsize=14)
        
        # Set a main title for the figure
        fig.suptitle(f'Subject {subject}', fontsize=20)
        
        # Adjust layout
        plt.subplots_adjust(left=0.05, right=0.9, top=0.85, bottom=0.1, wspace=0.2)
        
        # Show the figure
        plt.show()

# Assuming band_power_data is the output of your band power computation function
plot_tsne_band_power(band_power_data)



#%% 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# LDA model
lda = LinearDiscriminantAnalysis()

# Dictionary to store average cross-session accuracies for each subject
subject_averages = {}
all_subjects_accuracies = []

# Iterate over each subject in band_power_data
for subject, sessions in band_power_data.items():
    print(f"Subject: {subject}")
    
    num_sessions = len(sessions)
    # Initialize a matrix to store accuracies (num_sessions x num_sessions)
    session_accuracies = np.zeros((num_sessions, num_sessions))
    
    # List of session IDs (keys)
    session_ids = list(sessions.keys())
    
    # Train on one session and test on another
    for train_session_idx, train_session_id in enumerate(session_ids):
        # Training data
        train_data = sessions[train_session_id]['data']  # Shape: Trials x Channels x Time Windows x Bands
        train_labels = sessions[train_session_id]['label']  # Corresponding labels
        
        # Reshape train_data to (num_samples, num_features)
        num_trials, num_channels, num_windows, num_bands = train_data.shape
        x_train = train_data.transpose(0, 2, 1, 3).reshape(-1, num_channels * num_bands)  # Combine trials and windows
        y_train = np.repeat(train_labels, num_windows)  # Match labels to reshaped data
        
        # Train the LDA model
        lda.fit(x_train, y_train)
        
        # Test on all other sessions
        for test_session_idx, test_session_id in enumerate(session_ids):
            # Skip same-session pairs
            if train_session_id == test_session_id:
                continue
            
            # Testing data
            test_data = sessions[test_session_id]['data']  # Shape: Trials x Channels x Time Windows x Bands
            test_labels = sessions[test_session_id]['label']  # Corresponding labels
            
            # Reshape test_data to (num_samples, num_features)
            x_test = test_data.transpose(0, 2, 1, 3).reshape(-1, num_channels * num_bands)  # Combine trials and windows
            y_test = np.repeat(test_labels, num_windows)  # Match labels to reshaped data
            
            # Predict on the test session
            y_pred = lda.predict(x_test)
            
            # Calculate accuracy for the current test session
            accuracy = accuracy_score(y_test, y_pred)
            session_accuracies[train_session_idx, test_session_idx] = accuracy
            print(f"Accuracy (train on session {train_session_id}, test on session {test_session_id}): {accuracy:.4f}")

            # Store accuracy in the dataset for plotting
            all_subjects_accuracies.append({
                'Subject': subject,
                'Train-Test Pair': f'{train_session_id}->{test_session_id}',
                'Accuracy': accuracy
            })

    # Calculate the average cross-session accuracy for the subject
    # Exclude diagonal elements (training on the same session)
    cross_session_accuracies = session_accuracies[~np.eye(num_sessions, dtype=bool)]
    average_accuracy = np.mean(cross_session_accuracies)
    subject_averages[subject] = average_accuracy
    
    # Print the pairwise accuracy matrix for the subject
    print(f"\nPairwise session accuracies for Subject {subject}:")
    print(session_accuracies)
    print(f"Average cross-session accuracy for Subject {subject}: {average_accuracy:.4f}\n")

# Summary of average accuracies across all subjects
print("\nSummary of Cross-Session Average Accuracies:")
for subject, avg_accuracy in subject_averages.items():
    print(f"Subject {subject}: {avg_accuracy:.4f}")

# Create DataFrame for plotting
df = pd.DataFrame(all_subjects_accuracies)

# Box Plot: Accuracy Distribution Across Subjects
plt.figure(figsize=(14, 8))
sns.boxplot(x='Subject', y='Accuracy', data=df)
plt.title('Accuracy Distribution Across Subjects')
plt.xlabel('Subject')
plt.ylabel('Accuracy')
plt.xticks(rotation=90)  # Rotate x-axis labels if needed
plt.tight_layout()
plt.show()

# Modify the Train-Test Pair to be in the format '1v2', '1v3', etc.
df['Train-Test Pair'] = df['Train-Test Pair'].apply(lambda x: f"{x.split('->')[0]}v{x.split('->')[1]}")

# Pivot data for heatmap
heatmap_data = df.pivot(index='Subject', columns='Train-Test Pair', values='Accuracy')

# Plot the heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title('Accuracy Heatmap Across Subjects and Train-Test Pairs')
plt.xlabel('Train-Test Pair')
plt.ylabel('Subject')
plt.xticks(ticks=range(len(heatmap_data.columns)), labels=heatmap_data.columns, rotation=90)
plt.yticks(ticks=range(len(heatmap_data.index)), labels=heatmap_data.index, rotation=0)
plt.tight_layout()
plt.show()


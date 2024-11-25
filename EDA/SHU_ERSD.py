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
from scipy.signal import welch
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif
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
from scipy.signal import welch, cwt, morlet
from scipy.signal import butter, filtfilt
from scipy.io import loadmat

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


#%%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Function to compute ERD/ERS
def compute_erd_ers(data, fs=250, band=(8, 30), electrodes=[12, 11, 13]):
    """
    Computes the ERD/ERS for a given data sample and normalizes it to the range [-1, 1].
    
    Parameters:
    - data: The EEG data for a trial (shape: n_tsteps x n_electrodes)
    - fs: Sampling frequency (default: 250 Hz)
    - band: Frequency band for ERD/ERS calculation (default: (8, 30) Hz for alpha/beta)
    - electrodes: List of selected electrodes (default: [8, 9, 10] for C3, Cz, C4)
    
    Returns:
    - erd_ers: The normalized ERD/ERS values for each time step (shape: n_electrodes x num_freq_bins x steps)
    - freq: The frequency values corresponding to the bands (in Hz)
    """
    if data.shape[0] < fs:
        print("Warning: Data length is shorter than the sampling frequency. Skipping computation.")
        return None, None

    # Subset data to only the selected electrodes
    data = data[:, electrodes]
    data = np.transpose(data)  # Transpose to shape (n_electrodes, n_tsteps)

    # Find baseline power (over the first fs samples)
    freq, psd = signal.welch(data[:, :fs], fs, nperseg=fs)
    band_idx = np.where((freq >= band[0]) & (freq <= band[1]))[0]
    baseline = psd[:, band_idx]  # Mean PSD for each electrode over baseline period

    # Calculate the number of steps (third dimension size)
    n_tsteps = data.shape[1]  # Number of time steps
    steps = (n_tsteps - fs) // fs  # Number of full windows

    # Preallocate the ERD/ERS array
    erd_ers = np.zeros((len(electrodes), len(band_idx), steps))  # Shape (3, num_freq_bins, steps)

    # Compute ERD/ERS for each fs window
    step_idx = 0
    for idx in range(fs, n_tsteps, fs):
        if n_tsteps - idx < fs:  # Skip if remaining data is less than one window
            break

        # Extract current window of data
        x = data[:, idx:idx + fs]
        
        # Compute the PSD for the current window
        freq, psd = signal.welch(x, fs, nperseg=fs)
        
        # Select the frequency band of interest
        band_idx = np.where((freq >= band[0]) & (freq <= band[1]))[0]
        freq = freq[band_idx]
        current = psd[:, band_idx] 
        
        # Compute ERD/ERS
        final = (current - baseline) / baseline  # ERD/ERS for each time step
        
        # Normalize the ERD/ERS to the range [-1, 1]
        final_normalized = 2 * (final - np.min(final)) / (np.max(final) - np.min(final)) - 1
        
        # Store the result in the preallocated ERD/ERS array
        erd_ers[:, :, step_idx] = final_normalized
        step_idx += 1
    
    return erd_ers, freq

# Function to plot ERD/ERS
def plot_erd_ers(erd_ers, freq, fs=250, title="ERD/ERS Heatmap"):
    """
    Plots the ERD/ERS heatmap for each electrode (C3, Cz, C4) in separate subplots.
    
    Parameters:
    - erd_ers: The ERD/ERS values (3 electrodes, 12 frequency bands, timesteps)
    - freq: The frequency values corresponding to the frequency bands
    - fs: The sampling frequency (default: 250 Hz)
    - title: The title for the heatmap plot
    """
    # Define electrode labels
    electrode_labels = ['C3', 'Cz', 'C4']

    # Create subplots (3 columns for each electrode)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, dpi=300)  # High-resolution
    fig.suptitle(title, fontsize=16)

    # Loop through each electrode (C3, Cz, C4) and plot the ERD/ERS
    for i, ax in enumerate(axes):
        im = ax.imshow(erd_ers[i, :, :], aspect='auto', cmap='jet', origin='lower', interpolation='gaussian')
        ax.set_title(f"{electrode_labels[i]}")
        ax.set_xlabel("Time Steps (s)")
        ax.set_ylabel("Frequency Bands (Hz)")
        
        # Set y-ticks to correspond to frequency bands
        ax.set_yticks(np.arange(erd_ers.shape[1]))  
        ax.set_yticklabels(np.round(freq, 2))  # Use rounded frequency bands for y-axis labels

        # Set x-ticks for time steps
        ax.set_xticks(np.arange(erd_ers.shape[2]))  
        ax.set_xticklabels(np.arange(1, erd_ers.shape[2] + 1))  # Time steps in seconds

        # Add color bar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('ERD/ERS')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for spacing
    plt.show()

# Wrapper to process and plot trials for a specific subject
def process_and_plot_subject(filtered_data_split, subject_id, fs=250, band=(8, 30), electrodes=[12, 11, 13]):
    """
    Processes and plots ERD/ERS for a specified subject in the filtered_data_split dictionary.
    
    Parameters:
    - filtered_data_split: Dictionary containing subjects and their trial data
    - subject_id: The ID of the subject to process (e.g., 'sub-001')
    - fs: Sampling frequency (default: 250 Hz)
    - band: Frequency band for ERD/ERS computation (default: (8, 30))
    - electrodes: List of selected electrodes (default: [12, 11, 13] for C3, Cz, C4)
    """
    if subject_id not in filtered_data_split:
        print(f"Subject {subject_id} not found in the dataset.")
        return
    
    # Extract data for the specific subject
    conditions = filtered_data_split[subject_id]
    
    for condition, trials in conditions.items():
        for trial_idx, trial_data in enumerate(trials):
            # Compute ERD/ERS
            trial_data = np.transpose(trial_data)
            erd_ers, freq = compute_erd_ers(trial_data, fs=fs, band=band, electrodes=electrodes)
            if erd_ers is not None:
                # Plot the ERD/ERS heatmap
                title = f"ERD/ERS - Subject: {subject_id}, Condition: {condition}, Trial: {trial_idx + 1}"
                plot_erd_ers(erd_ers, freq, fs=fs, title=title)

# Example usage with a specific subject ID
process_and_plot_subject(filtered_data_split, subject_id='sub-016', fs=250, band=(8, 30), electrodes=[12, 11, 13])

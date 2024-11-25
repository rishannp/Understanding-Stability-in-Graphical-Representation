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


def compute_erd_ers(data, fs=256, band=(8, 30), electrodes=[8, 9, 10]):
    """
    Computes the ERD/ERS for a given data sample and normalizes it to the range [-1, 1].
    
    Parameters:
    - data: The EEG data for a trial (shape: n_tsteps x n_electrodes)
    - fs: Sampling frequency (default: 256 Hz)
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

    # Find baseline power (over the first fs//2 samples)
    freq, psd = signal.welch(data[:, :fs], fs, nperseg=fs)
    band_idx = np.where((freq >= band[0]) & (freq <= band[1]))[0]
    baseline = psd[:, band_idx]  # Mean PSD for each electrode over baseline period

    # Calculate the number of steps (third dimension size)
    n_tsteps = data.shape[1]  # Number of time steps
    steps = (n_tsteps - fs) // (fs)  # Number of full windows

    # Preallocate the ERD/ERS array
    erd_ers = np.zeros((len(electrodes), len(band_idx), steps))  # Shape (3, num_freq_bins, steps)

    # Initialize a counter for the third dimension
    step_idx = 0

    # Find current power for each fs//2 window and compute ERD/ERS
    for idx in range(fs, n_tsteps, fs):
        # Check if the remaining data is smaller than fs//2, then exit loop
        if n_tsteps - idx < fs:
            break
        
        # Extract current window of data (of length fs//2)
        x = data[:, idx:idx + fs]
        
        # Compute the PSD for the current window
        freq, psd = signal.welch(x, fs, nperseg=fs)
        
        # Select the frequency band of interest
        band_idx = np.where((freq >= band[0]) & (freq <= band[1]))[0]
        freq = freq[band_idx]
        current = psd[:, band_idx] 
        
        # Subtract baseline power element-wise and divide by baseline power element-wise
        final = (current - baseline) / baseline  # ERD/ERS for each time step
        
        # Normalize the ERD/ERS to the range [-1, 1]
        final_normalized = 2 * (final - np.min(final)) / (np.max(final) - np.min(final)) - 1
        
        # Store the result in the preallocated ERD/ERS array
        erd_ers[:, :, step_idx] = final_normalized
        
        # Increment the step index
        step_idx += 1
    
    return erd_ers, freq


def plot_erd_ers(erd_ers, freq, fs=256, title="ERD/ERS Heatmap"):
    """
    Plots the ERD/ERS heatmap for each electrode (C3, Cz, C4) in separate subplots.
    
    Parameters:
    - erd_ers: The ERD/ERS values (3 electrodes, 12 frequency bands, timesteps)
    - freq: The frequency values corresponding to the frequency bands
    - fs: The sampling frequency (default: 256 Hz)
    - title: The title for the heatmap plot
    """
    # Define electrode labels
    electrode_labels = ['C3', 'Cz', 'C4']

    # Create subplots (3 columns for each electrode)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, dpi=300)  # Set dpi to increase resolution
    fig.suptitle(title, fontsize=16)

    # Loop through each electrode (C3, Cz, C4) and plot the ERD/ERS
    for i, ax in enumerate(axes):
        # Use 'gaussian' or 'bicubic' interpolation for smoother visualization
        im = ax.imshow(erd_ers[i, :, :], aspect='auto', cmap='jet', origin='lower', interpolation='gaussian')
        ax.set_title(f"{electrode_labels[i]}")
        ax.set_xlabel("Time Steps (s)")
        ax.set_ylabel("Frequency Bands (Hz)")
        
        # Set y-ticks to correspond to frequency bands
        ax.set_yticks(np.arange(erd_ers.shape[1]))  # Set y-ticks to match frequency bands
        ax.set_yticklabels(freq[0:erd_ers.shape[1]])  # Use frequency bands for y-axis labels

        # Adjust x-ticks to represent time in seconds, starting from 0.5, 1.0, ...
        time_steps = np.arange(1, erd_ers.shape[2] * 1 + 1, 1)
        ax.set_xticks(np.arange(erd_ers.shape[2]))  # Set x-ticks to match time steps
        ax.set_xticklabels(time_steps)  # Label x-ticks with time in seconds

        # Add color bar for the current subplot
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('ERD/ERS')

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

#%%



# % Preparing Data
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Penn State Data\Work\Data\OG_Full_Data'

# Define the subject numbers: subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
subject_numbers = [9]

# Dictionary to hold the loaded data for each subject
subject_data = {}
all_subjects_accuracies = {}

# Loop through the subject numbers and load the corresponding data
for subject_number in subject_numbers:
    mat_fname = pjoin(data_dir, f'OGFS{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_data[f'S{subject_number}'] = mat_contents[f'Subject{subject_number}']
    S1 = subject_data[f'S{subject_number}'][:,:]

    for idx in ['L', 'R']:  # Iterate through the conditions (L and R)
        trials = S1[idx].shape[1]-1  # Get the number of trials for this condition
        
        # Loop through each trial for the current condition
        for trial_idx in range(trials):
            print(f"Processing Subject {subject_number} - {idx} Trial {trial_idx+1}...")
            
            # Compute the ERD/ERS for the trial, along with the frequency values
            erd_ers, freq = compute_erd_ers(S1[idx][0, trial_idx], fs=256, band=(8, 30), electrodes=[8, 9, 10])
            
            # Plot the ERD/ERS heatmap for this trial
            plot_erd_ers(erd_ers, freq, fs=256, title=f"Subject {subject_number} {idx} Trial {trial_idx+1} ERD/ERS Heatmap")


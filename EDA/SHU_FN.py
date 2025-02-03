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
import os
import scipy
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.fftpack import fft, ifft
import mne
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
from sklearn.metrics import accuracy_score
from torch_geometric.data import Data

channels = np.array([
    'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5',
    'FC6', 'CZ', 'C3', 'C4', 'T3', 'T4', 'A1', 'A2', 'CP1', 'CP2',
    'CP5', 'CP6', 'PZ', 'P3', 'P4', 'T5', 'T6', 'PO3', 'PO4', 'OZ',
    'O1', 'O2'
])

# Define the directory containing your .mat files
directory = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\SHU Dataset\MatFiles'

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

# Merge all five folds for each subject separately for 'L' and 'R' trials
merged_subject_data = {}

for subject_id, sessions in merged_fold_data.items():
    # Retrieve data and labels for all folds, separately for L and R
    all_L_data = [sessions[fold_id]['data'][sessions[fold_id]['label'] == 0] for fold_id in range(5)]
    all_R_data = [sessions[fold_id]['data'][sessions[fold_id]['label'] == 1] for fold_id in range(5)]
    
    # Concatenate along the trials axis for L and R separately
    merged_L_data = np.concatenate(all_L_data, axis=0)  # Shape: (Total L Trials, Channels, Samples)
    merged_R_data = np.concatenate(all_R_data, axis=0)  # Shape: (Total R Trials, Channels, Samples)

    # Store in the new merged structure
    merged_subject_data[subject_id] = {
        'L': merged_L_data,
        'R': merged_R_data
    }


#%%

import numpy as np
import scipy.io as sio
import torch
from scipy.signal import butter, filtfilt, hilbert, coherence
from scipy.linalg import norm
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
from joblib import Parallel, delayed

# ---------------------------
# Signal Processing Functions
# ---------------------------
def bandpass_filter(data, freq_range, sample_rate=256, order=5):
    """Applies a bandpass Butterworth filter."""
    nyquist = 0.5 * sample_rate
    low, high = freq_range[0] / nyquist, freq_range[1] / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def compute_cfc(eeg_data, fs=256):
    """Computes cross-frequency coupling (CFC) using PAC."""
    mu_data = bandpass_filter(eeg_data, [8, 13], fs)
    beta_data = bandpass_filter(eeg_data, [13, 30], fs)
    mu_phase = np.angle(hilbert(mu_data, axis=-1))
    beta_amp = np.abs(hilbert(beta_data, axis=-1))

    pac_matrix = np.abs(np.mean(beta_amp[:, None, :] * np.exp(1j * mu_phase[None, :, :]), axis=-1))
    return (pac_matrix - pac_matrix.min()) / (pac_matrix.max() - pac_matrix.min() + 1e-10)  

def compute_msc(eeg_data, fs=256, nperseg=256):
    """Computes mean squared coherence (MSC)."""
    num_electrodes = eeg_data.shape[0]
    msc_matrix = np.zeros((num_electrodes, num_electrodes))

    for i in range(num_electrodes):
        for j in range(i + 1, num_electrodes):
            _, Cxy = coherence(eeg_data[i], eeg_data[j], fs=fs, nperseg=nperseg)
            msc_matrix[i, j] = msc_matrix[j, i] = np.mean(Cxy)

    return msc_matrix

def compute_cmi(eeg_data):
    """Computes conditional mutual information (CMI)."""
    num_electrodes = eeg_data.shape[0]
    cmi_matrix = np.zeros((num_electrodes, num_electrodes))

    for i in range(num_electrodes):
        for j in range(i + 1, num_electrodes):
            cmi_value = mutual_info_regression(eeg_data[i].reshape(-1, 1), eeg_data[j])[0]
            cmi_matrix[i, j] = cmi_matrix[j, i] = cmi_value

    return cmi_matrix

def compute_plv(eeg_data):
    """Computes Phase Locking Value (PLV)."""
    num_electrodes = eeg_data.shape[0]
    phase = np.angle(hilbert(eeg_data, axis=-1))
    phase_diff = np.exp(1j * (phase[:, None, :] - phase[None, :, :]))
    return np.abs(np.mean(phase_diff, axis=-1))

# ---------------------------
# Utility Functions
# ---------------------------

def frobenius_norm(matrix1, matrix2):
    """Computes Frobenius norm between two matrices."""
    return norm(matrix1 - matrix2, 'fro')

def compute_connectivity_matrices(eeg_data, fs=250):
    """Computes all connectivity matrices for a single EEG trial."""
    return {
        'PLV': compute_plv(eeg_data),
        'MSC': compute_msc(eeg_data, fs),
        'CFC': compute_cfc(eeg_data, fs),
        'CMI': compute_cmi(eeg_data)
    }

def process_subject(subject_data, fs=250):
    """Parallelized function to compute connectivity matrices for all trials in a subject."""
    num_trials = subject_data.shape[0]
    return Parallel(n_jobs=-1)(delayed(compute_connectivity_matrices)(subject_data[i], fs) for i in range(num_trials))

# ---------------------------
# Compute Successive and Inter-Measure Frobenius Differences
# ---------------------------

successive_metrics = {measure: {'L': [], 'R': []} for measure in ['PLV', 'MSC', 'CFC', 'CMI']}
inter_measure_metrics = {pair: {'L': [], 'R': []} for pair in ["PLV_MSC", "PLV_CFC", "PLV_CMI", "MSC_CFC", "MSC_CMI", "CFC_CMI"]}

for subject in tqdm(merged_subject_data.keys(), desc="Processing Subjects"):
    for side in ['L', 'R']:  
        subject_data = merged_subject_data[subject][side]

        if subject_data.shape[0] < 2:
            continue  

        # Compute connectivity matrices per trial
        conn_matrices = Parallel(n_jobs=-1)(delayed(compute_connectivity_matrices)(trial) for trial in subject_data)

        # Convert to dictionary format with stacked matrices
        conn_stacked = {measure: np.stack([trial[measure] for trial in conn_matrices])
                        for measure in ['PLV', 'MSC', 'CFC', 'CMI']}

        # Compute successive Frobenius norm differences
        for measure in ['PLV', 'MSC', 'CFC', 'CMI']:
            diffs = [frobenius_norm(conn_stacked[measure][t], conn_stacked[measure][t + 1])
                     for t in range(conn_stacked[measure].shape[0] - 1)]
            successive_metrics[measure][side].append(np.mean(diffs))  

        # Compute inter-measure Frobenius differences (per trial)
        for t in range(len(conn_matrices)):
            inter_measure_metrics["PLV_MSC"][side].append(frobenius_norm(conn_matrices[t]["PLV"], conn_matrices[t]["MSC"]))
            inter_measure_metrics["PLV_CFC"][side].append(frobenius_norm(conn_matrices[t]["PLV"], conn_matrices[t]["CFC"]))
            inter_measure_metrics["PLV_CMI"][side].append(frobenius_norm(conn_matrices[t]["PLV"], conn_matrices[t]["CMI"]))
            inter_measure_metrics["MSC_CFC"][side].append(frobenius_norm(conn_matrices[t]["MSC"], conn_matrices[t]["CFC"]))
            inter_measure_metrics["MSC_CMI"][side].append(frobenius_norm(conn_matrices[t]["MSC"], conn_matrices[t]["CMI"]))
            inter_measure_metrics["CFC_CMI"][side].append(frobenius_norm(conn_matrices[t]["CFC"], conn_matrices[t]["CMI"]))

# ---------------------------
# Compute Class-Wise Averages
# ---------------------------

summary_successive = {measure: {side: np.mean(successive_metrics[measure][side]) for side in ['L', 'R']}
                      for measure in ['PLV', 'MSC', 'CFC', 'CMI']}

summary_inter_measure = {pair: {side: np.mean(inter_measure_metrics[pair][side]) for side in ['L', 'R']}
                         for pair in ["PLV_MSC", "PLV_CFC", "PLV_CMI", "MSC_CFC", "MSC_CMI", "CFC_CMI"]}

# ---------------------------
# Print Results
# ---------------------------

import pandas as pd

df_successive = pd.DataFrame(summary_successive).T
df_inter_measure = pd.DataFrame(summary_inter_measure).T

print("\nSuccessive Frobenius Norm Differences:")
print(df_successive)

print("\nInter-Measure Frobenius Norm Differences:")
print(df_inter_measure)

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
import time
from torch_geometric.explain import Explainer, GNNExplainer
import seaborn as sns

#% Functions
def plvfcn(eegData):
    numElectrodes = eegData.shape[1]
    numTimeSteps = eegData.shape[0]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(sig.hilbert(eegData[:, electrode1]))
            phase2 = np.angle(sig.hilbert(eegData[:, electrode2]))
            phase_difference = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_difference)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    return plvMatrix

def compute_plv(subject_data):
    idx = ['L', 'R']
    numElectrodes = subject_data['L'][0,1].shape[1]
    plv = {field: np.zeros((numElectrodes, numElectrodes, subject_data.shape[1])) for field in idx}
    for i, field in enumerate(idx):
        for j in range(subject_data.shape[1]):
            x = subject_data[field][0, j]
            plv[field][:, :, j] = plvfcn(x)
    l, r = plv['L'], plv['R']
    yl, yr = np.zeros((subject_data.shape[1], 1)), np.ones((subject_data.shape[1], 1))
    img = np.concatenate((l, r), axis=2)
    y = np.concatenate((yl, yr), axis=0)
    y = torch.tensor(y, dtype=torch.long)
    return img, y


def create_graphs(plv, threshold):
    
    graphs = []
    for i in range(plv.shape[2]):
        G = nx.Graph()
        G.add_nodes_from(range(plv.shape[0]))
        for u in range(plv.shape[0]):
            for v in range(plv.shape[0]):
                if u != v and plv[u, v, i] > threshold:
                    G.add_edge(u, v, weight=plv[u, v, i])
        graphs.append(G)
    return graphs


def aggregate_eeg_data(S1,band): #%% This is to get the feat vector
    """
    Aggregate EEG data for each class.

    Parameters:
        S1 (dict): Dictionary containing EEG data for each class. Keys are class labels, 
                   values are arrays of shape (2, num_samples, num_channels), where the first dimension
                   corresponds to EEG data (index 0) and frequency data (index 1).

    Returns:
        l (ndarray): Aggregated EEG data for class 'L'.
        r (ndarray): Aggregated EEG data for class 'R'.
    """
    idx = ['L', 'R']
    numElectrodes = S1['L'][0,1].shape[1];
    max_sizes = {field: 0 for field in idx}

    # Find the maximum size of EEG data for each class
    for field in idx:
        for i in range(S1[field].shape[1]):
            max_sizes[field] = max(max_sizes[field], S1[field][0, i].shape[0])

    # Initialize arrays to store aggregated EEG data
    l = np.zeros((max_sizes['L'], numElectrodes, S1['L'].shape[1]))
    r = np.zeros((max_sizes['R'], numElectrodes, S1['R'].shape[1]))

    # Loop through each sample
    for i in range(S1['L'].shape[1]):
        for j, field in enumerate(idx):
            x = S1[field][0, i]  # EEG data for the current sample
            # Resize x to match the maximum size
            resized_x = np.zeros((max_sizes[field], 22))
            resized_x[:x.shape[0], :] = x
            # Add the resized EEG data to the respective array
            if field == 'L':
                l[:, :, i] += resized_x
            elif field == 'R':
                r[:, :, i] += resized_x

    l = l[..., np.newaxis]
    l = np.copy(l) * np.ones(len(band)-1)

    r = r[..., np.newaxis]
    r = np.copy(r) * np.ones(len(band)-1)
    
    return l, r

def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = sig.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sig.sosfiltfilt(sos, data)
    return filtered_data


def bandpower(data,low,high):

    fs = 256
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


# % Preparing Data
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Penn State Data\Work\Data\OG_Full_Data'

# Define the subject numbers
subject_numbers = [1,2,5,9,21,31,34,39]

# Dictionary to hold the loaded data for each subject
subject_data = {}
all_subjects_accuracies = {}

# Loop through the subject numbers and load the corresponding data
for subject_number in subject_numbers:
    mat_fname = pjoin(data_dir, f'OGFS{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_data[f'S{subject_number}'] = mat_contents[f'Subject{subject_number}']
    S1 = subject_data[f'S{subject_number}'][:,:]
    
    plv, y = compute_plv(S1)

import os
import seaborn as sns
import matplotlib.pyplot as plt

# Base directory where plots will be saved
base_save_dir = r"C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Understanding Stability in Graphical Representation\Plots\PLVPLot"

# Loop through each subject
for subject_number in subject_numbers:
    subject_id = f"S{subject_number}"
    subject_save_dir = os.path.join(base_save_dir, subject_id)
    
    # Create subject-specific directory if it doesn't exist
    os.makedirs(subject_save_dir, exist_ok=True)
    
    # Loop through trials for the subject
    plv, y = compute_plv(subject_data[subject_id])
    
    for trial_idx in range(plv.shape[2]):
        trial_data = plv[:, :, trial_idx]
        trial_type = "L" if trial_idx < plv.shape[2] // 2 else "R"  # Determine Left or Right trial
        
        # Create subfolder for the trial type ("L" or "R")
        trial_save_dir = os.path.join(subject_save_dir, trial_type)
        os.makedirs(trial_save_dir, exist_ok=True)
        
        # Create the heatmap plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(trial_data, cmap='jet', cbar=True, square=True, vmin=0, vmax=np.max(plv))
        plt.title(f"Trial {trial_idx + 1} ({trial_type})")
        plt.xlabel("Electrodes")
        plt.ylabel("Electrodes")
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(trial_save_dir, f"Trial_{trial_idx + 1}.png")
        plt.savefig(save_path)
        plt.close()  # Close the plot to free memory
        
        
#%% SHU

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


def plvfcn(eegData):
    numElectrodes = eegData.shape[0]
    numTimeSteps = eegData.shape[1]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(sig.hilbert(eegData[electrode1, :]))
            phase2 = np.angle(sig.hilbert(eegData[electrode2, :]))
            phase_difference = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_difference)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    return plvMatrix


# Function to create graphs and compute adjacency matrices for PLV matrices
def create_graphs_and_edges(plv_matrices, threshold):
    graphs = []
    adj_matrices = np.zeros([plv_matrices.shape[0], plv_matrices.shape[0], plv_matrices.shape[2]])  # (Electrodes, Electrodes, Trials)
    edge_indices = []
    
    for i in range(plv_matrices.shape[2]):
        G = nx.Graph()
        G.add_nodes_from(range(plv_matrices.shape[0]))  # Nodes represent electrodes
        
        # Initialize lists for storing adjacency matrix and edge indices
        source_nodes = []
        target_nodes = []
        
        # Iterate over electrode pairs to construct the graph and adjacency matrix
        for u in range(plv_matrices.shape[0]):
            for v in range(u + 1, plv_matrices.shape[0]):  # Avoid duplicate edges
                if plv_matrices[u, v, i] > threshold:
                    # Add edge to graph
                    G.add_edge(u, v, weight=plv_matrices[u, v, i])
                    
                    # Store the edge in adjacency matrix (symmetric)
                    adj_matrices[u, v, i] = plv_matrices[u, v, i]
                    adj_matrices[v, u, i] = plv_matrices[u, v, i]
                    
                    # Store source and target nodes for edge indices
                    source_nodes.append(u)
                    target_nodes.append(v)
        
        # Convert adjacency matrix and graphs
        graphs.append(G)
        
        # Convert the lists to a LongTensor for edge index
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_indices.append(edge_index)
    
    # Convert adjacency matrices to torch tensors
    adj_matrices = torch.tensor(adj_matrices, dtype=torch.float32)
    
    # Stack edge indices for all trials (session-wise)
    edge_indices = torch.stack(edge_indices, dim=-1)
    
    return adj_matrices, edge_indices, graphs


# Function to compute PLV matrices for all trials in all sessions of a subject
def compute_plv(subject_data):
    session_plv_data = {}  # To store PLV matrices, labels, and graphs for each session
    
    # Iterate over each session in the subject's data
    for session_id, session_data in subject_data.items():
        data = session_data['data']  # Shape: (Trials, Channels, TimeSteps)
        labels = session_data['label']  # Shape: (Trials,)
        
        numTrials, numElectrodes, _ = data.shape
        plv_matrices = np.zeros((numElectrodes, numElectrodes, numTrials))
        
        # Compute PLV for each trial in the session
        for trial_idx in range(numTrials):
            eeg_trial_data = data[trial_idx]  # Shape: (Channels, TimeSteps)
            plv_matrices[:, :, trial_idx] = plvfcn(eeg_trial_data)
        
        # Convert labels to torch tensor
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Create graphs, adjacency matrices, and edge indices
        adj_matrices, edge_indices, graphs = create_graphs_and_edges(plv_matrices, threshold=0)
        
        # Store session-level data for PLV matrices, labels, and graphs
        session_plv_data[session_id] = {
            'plv_matrices': plv_matrices,   # Shape: (Electrodes, Electrodes, Trials)
            'labels': label_tensor,         # Shape: (Trials,)
            'adj_matrices': adj_matrices,   # Shape: (Electrodes, Electrodes, Trials)
            'edge_indices': edge_indices,   # Shape: (2, Edges, Trials)
            'graphs': graphs                # List of graphs for each trial
        }
    
    return session_plv_data

# Dictionary to store PLV data for all subjects
subject_plv_data = {}

# Loop over each subject in the dataset
for subject_id, subject_data in merged_fold_data.items():
    print(f"Processing subject: {subject_id}")
    
    # Initialize subject entry in dictionary
    subject_plv_data[subject_id] = {}
    
    # Compute PLV matrices, labels, and graph-related data for each session of the subject
    session_data = compute_plv(subject_data)
    
    # Store the computed data for each session in the subject's data dictionary
    subject_plv_data[subject_id] = session_data
    
#%%


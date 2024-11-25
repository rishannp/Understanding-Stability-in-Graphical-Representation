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

#%% Functions
def cmifcn(eegData):
    numElectrodes = eegData.shape[1]
    cmiMatrix = np.zeros((numElectrodes, numElectrodes))
    
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            # Extract the signals for the two electrodes
            signal1 = eegData[:, electrode1]
            signal2 = eegData[:, electrode2]
            
            # Compute mutual information between the two signals
            cmi_value = mutual_info_regression(signal1.reshape(-1, 1), signal2)
            
            # Store the MI value in the matrix (symmetrically)
            cmiMatrix[electrode1, electrode2] = cmi_value
            cmiMatrix[electrode2, electrode1] = cmi_value
    
    return cmiMatrix

def compute_cmi(subject_data):
    idx = ['L', 'R']
    numElectrodes = subject_data['L'][0, 1].shape[1]
    cmi = {field: np.zeros((numElectrodes, numElectrodes, subject_data[field].shape[1])) for field in idx}

    for i, field in enumerate(idx):
        for j in range(subject_data[field].shape[1]):
            x = subject_data[field][0, j]
            cmi[field][:, :, j] = cmifcn(x)
    
    # Combine results for 'L' and 'R'
    l, r = cmi['L'], cmi['R']
    yl, yr = np.zeros((subject_data['L'].shape[1], 1)), np.ones((subject_data['R'].shape[1], 1))
    img = np.concatenate((l, r), axis=2)
    y = np.concatenate((yl, yr), axis=0)
    
    # Convert to torch tensor
    y = torch.tensor(y, dtype=torch.long)
    
    return img, y

def create_graphs(cfc, threshold):
    graphs = []
    for i in range(cfc.shape[2]):
        G = nx.Graph()
        G.add_nodes_from(range(cfc.shape[0]))
        for u in range(cfc.shape[0]):
            for v in range(cfc.shape[0]):
                if u != v and cfc[u, v, i] > threshold:
                    G.add_edge(u, v, weight=cfc[u, v, i])
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


def bandpasscfc(data, freq_range, sample_rate=256, order=5):
    nyquist = 0.5 * sample_rate
    low = freq_range[0] / nyquist
    high = freq_range[1] / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=0)
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

#%%

# Directory containing the data
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Penn State Data\Work\Data\OG_Full_Data'

# Define the subject numbers
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]

# Dictionary to hold the loaded data for each subject
subject_data = {}
subject_results = {}  # Dictionary to store edge_index and y

# Loop through the subject numbers and load the corresponding data
for subject_number in subject_numbers:
    mat_fname = pjoin(data_dir, f'OGFS{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_data[f'S{subject_number}'] = mat_contents[f'Subject{subject_number}']
    S1 = subject_data[f'S{subject_number}'][:,:]
    S1 = S1[:, :-1]
    
    # Compute CFC and create graphs
    cfc, y = compute_cmi(S1)
    threshold = 0
    graphs = create_graphs(cfc, threshold)
    numElectrodes = S1['L'][0,1].shape[1]
    
    # Initialize the adjacency matrices
    adj = np.zeros([numElectrodes, numElectrodes, len(graphs)])
    for i, G in enumerate(graphs):
        adj[:, :, i] = nx.to_numpy_array(G)
    
    # Initialize an empty list to store edge indices
    edge_indices = []

    # Iterate over the adjacency matrices
    for i in range(adj.shape[2]):
        # Initialize lists to store source and target nodes
        source_nodes = []
        target_nodes = []
        
        # Iterate through each element of the adjacency matrix
        for row in range(adj.shape[0]):
            for col in range(adj.shape[1]):
                # Check if there's an edge
                if adj[row, col, i] >= threshold:
                    # Add source and target nodes to the lists
                    source_nodes.append(row)
                    target_nodes.append(col)
        
        # Create edge index as a NumPy array
        edge_index = np.array([source_nodes, target_nodes])
        
        # Append edge index to the list
        edge_indices.append(edge_index)

    # Convert edge_indices list to a NumPy array (3D array: 2 x num_edges x num_graphs)
    edge_indices = np.stack(edge_indices, axis=-1)

    # Save the edge_index and y into the results dictionary
    subject_results[f'S{subject_number}'] = {
        'edge_index': edge_indices,
        'y': y
    }

    band = list(range(8, 41, 4))
    l, r = aggregate_eeg_data(S1, band)
    l, r = np.transpose(l, [1, 0, 2, 3]), np.transpose(r, [1, 0, 2, 3])
    fs = 256
    
    for i in range(l.shape[3]):
        bp = [band[i], band[i+1]]
        for j in range(l.shape[2]):
            l[:, :, j, i] = bandpass(l[:, :, j, i], bp, sample_rate=fs)
            r[:, :, j, i] = bandpass(r[:, :, j, i], bp, sample_rate=fs)
    
    # Calculate bandpower
    l = bandpowercalc(l, band, fs)
    r = bandpowercalc(r, band, fs)

    # Combine the left and right data
    x = np.concatenate([l, r], axis=2)

    # Convert x to a NumPy array (ensure it's already a NumPy array)
    x = np.array(x, dtype=np.float32)

    # Save x into the results dictionary under the same subject key
    subject_results[f'S{subject_number}']['x'] = x

# Clean up variables no longer needed
del col, edge_index, i, row, source_nodes, target_nodes, S1, adj, graphs, cfc, l, r, band, fs, bp

#%%


# Initialize dictionaries to store the split results
split_results = {}

for subject, data in subject_results.items():
    edge_index = data['edge_index']
    x = data['x']
    y = data['y']
    
    # Find indices for Left and Right trials
    left_indices = np.where(y == 0)[0]
    right_indices = np.where(y == 1)[0]
    
    # Determine number of trials per fold
    left_fold_size = len(left_indices) // 4
    right_fold_size = len(right_indices) // 4
    
    # Initialize lists to hold fold data
    left_folds = []
    right_folds = []
    
    # Split Left and Right trials into 4 folds
    for i in range(4):
        left_start = i * left_fold_size
        right_start = i * right_fold_size
        
        if i < 3:  # First 3 folds
            left_folds.append(left_indices[left_start:left_start + left_fold_size])
            right_folds.append(right_indices[right_start:right_start + right_fold_size])
        else:  # Last fold takes the remainder
            left_folds.append(left_indices[left_start:])
            right_folds.append(right_indices[right_start:])
    
    # Combine corresponding Left and Right folds
    combined_folds = []
    for i in range(4):
        combined_indices = np.concatenate((left_folds[i], right_folds[i]))
        
        # Extract corresponding edge_index, x, and y data
        combined_edge_index = edge_index[:, :, combined_indices]
        combined_x = x[:, :, combined_indices]
        combined_y = y[combined_indices]
        
        combined_folds.append({
            'edge_index': combined_edge_index,
            'x': combined_x,
            'y': combined_y
        })
    
    # Store the results for the subject
    split_results[subject] = combined_folds

# Now, `split_results` will have the desired data structure for each subject.

#%% 

from sklearn.manifold import TSNE

# Number of folds
num_folds = 4

# Iterate over each subject in split_results
for subject in split_results.keys():
    # Create a new figure for each subject
    fig, axes = plt.subplots(1, num_folds, figsize=(24, 8), sharex=True, sharey=True)  # Increase figsize for larger plots
    
    # Iterate over each fold
    for fold_idx in range(num_folds):
        # Extract the 'x' data and 'y' labels for the current fold
        fold_data = split_results[subject][fold_idx]
        x_data = fold_data['x']  # shape should be (num_features, num_samples)
        y_labels = fold_data['y']
        
        # Reshape x_data to have shape (num_samples, num_features)
        num_samples = x_data.shape[2]
        x_reshaped = x_data.reshape(-1, num_samples).T
        
        # Apply t-SNE to the reshaped data
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(x_reshaped)
        
        # Plot the t-SNE results
        ax = axes[fold_idx]
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_labels, cmap='viridis', s=20)  # Increase marker size
        ax.set_title(f'Fold {fold_idx + 1}', fontsize=16)  # Increase title fontsize
        ax.axis('off')  # Hide axis
    
    # Optionally, add a colorbar to the last subplot
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Labels', fontsize=14)  # Increase colorbar label fontsize
    
    # Set a main title for the figure
    fig.suptitle(f'Subject {subject}', fontsize=20)  # Increase main title fontsize
    
    # Adjust layout: reduce the space between subplots and leave space for the colorbar
    plt.subplots_adjust(left=0.05, right=0.9, top=0.85, bottom=0.1, wspace=0.2, hspace=0.2)
    
    # Show the figure
    plt.show()

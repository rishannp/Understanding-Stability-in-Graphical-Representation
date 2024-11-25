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

#% Functions

def bandpasscfc(data, freq_range, sample_rate=256, order=5):
    nyquist = 0.5 * sample_rate
    low = freq_range[0] / nyquist
    high = freq_range[1] / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


def cfcfcn(eegData, fs=256):
    numElectrodes = eegData.shape[1]
    cfcMatrix = np.zeros((numElectrodes, numElectrodes))

    # Bandpass filter the data for Mu (8-13 Hz) and Beta (13-30 Hz) bands
    mu_data = bandpasscfc(eegData, [8, 13], sample_rate=fs)
    beta_data = bandpasscfc(eegData, [13, 30], sample_rate=fs)

    # Compute the phase of the Mu band and the amplitude of the Beta band
    mu_phase = np.angle(hilbert(mu_data, axis=0))
    beta_amplitude = np.abs(hilbert(beta_data, axis=0))

    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            # Compute CFC using the phase of Mu and the amplitude of Beta
            pac_value = np.abs(np.mean(beta_amplitude[:, electrode2] * np.exp(1j * mu_phase[:, electrode1])))
            cfcMatrix[electrode1, electrode2] = pac_value
            cfcMatrix[electrode2, electrode1] = pac_value
    
    return cfcMatrix

def compute_cfc(subject_data, fs=256):
    idx = ['L', 'R']
    numElectrodes = subject_data['L'][0, 1].shape[1]
    cfc = {field: np.zeros((numElectrodes, numElectrodes, subject_data[field].shape[1])) for field in idx}

    for i, field in enumerate(idx):
        for j in range(subject_data[field].shape[1]):
            x = subject_data[field][0, j]
            cfc[field][:, :, j] = cfcfcn(x, fs=fs)

    l, r = cfc['L'], cfc['R']
    yl, yr = np.zeros((subject_data['L'].shape[1], 1)), np.ones((subject_data['R'].shape[1], 1))
    img = np.concatenate((l, r), axis=2)
    y = np.concatenate((yl, yr), axis=0)
    
    
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
    cfc, y = compute_cfc(S1)
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
        'y': y,
        'x': adj
    }

 

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

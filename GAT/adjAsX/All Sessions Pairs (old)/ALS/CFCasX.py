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

#%%



# % Preparing Data
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Penn State Data\Work\Data\OG_Full_Data'

# Define the subject numbers
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]

# Dictionary to hold the loaded data for each subject
subject_data = {}
all_subjects_accuracies = {}

# Loop through the subject numbers and load the corresponding data
for subject_number in subject_numbers:
    mat_fname = pjoin(data_dir, f'OGFS{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_data[f'S{subject_number}'] = mat_contents[f'Subject{subject_number}']
    S1 = subject_data[f'S{subject_number}'][:,:]
    
    plv, y = compute_cfc(S1)
    threshold = 0
    graphs = create_graphs(plv, threshold)
    numElectrodes = S1['L'][0,1].shape[1]
    adj = np.zeros([numElectrodes, numElectrodes, len(graphs)])
    for i, G in enumerate(graphs):
        adj[:, :, i] = nx.to_numpy_array(G)
    
    adj = torch.tensor(adj,dtype=torch.float32)
    #% Initialize an empty list to store edge indices
    edge_indices = [] # % Edge indices are a list of source and target nodes in a graph. Think of it like the adjacency matrix

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
                else:
                    # If no edge exists, add placeholder zeros to maintain size
                    source_nodes.append(0)
                    target_nodes.append(0)
        
        # Create edge index as a LongTensor
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        
        # Append edge index to the list
        edge_indices.append(edge_index)

    # Stack all edge indices along a new axis to create a 2D tensor
    edge_indices = torch.stack(edge_indices, dim=-1)

    del col,edge_index,i,row,source_nodes,target_nodes
    
    from torch_geometric.data import Data
    
    y = torch.tensor(y, dtype=torch.long)

    data_list = []
    for i in range(np.size(adj,2)):
        data_list.append(Data(x=adj[:, :, i], edge_index=edge_indices[:,:,i], y=y[i, 0]))
        
    datal =[]
    datar =[]
    size = len(data_list)
    idx = size//2
    c = [0,idx,idx*2,idx*3]

    datal = data_list[c[0]:c[1]]
    datar = data_list[c[1]:c[2]]

    data_list = []

    for i in range(idx):
        x = [datal[i],datar[i]] #datare[i]]
        data_list.extend(x)


    size = len(data_list)
    
    # Initialize a dictionary to store accuracies for each subject
    
    # Assuming 'data_list' is your data for the current subject
    num_folds = 4
    fold_size = len(data_list) // num_folds
    
    # Split the data manually into 4 folds
    folds = [data_list[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]
    
    # List to store fold accuracies for the current subject
    fold_accuracies = []
    
    # Iterate over each fold as training data
    for train_fold_idx in range(num_folds):
        # Iterate over each fold as testing data (excluding the current training fold)
        for test_fold_idx in range(num_folds):
            if test_fold_idx == train_fold_idx:
                continue  # Skip if the fold is the same as the training fold
    
            # Create training and testing sets
            train = folds[train_fold_idx]
            test = folds[test_fold_idx]
    
            # Set the random seed for reproducibility
            torch.manual_seed(12345)
    
            # Create DataLoader for train and test sets
            train_loader = DataLoader(train, batch_size=32, shuffle=False)
            test_loader = DataLoader(test, batch_size=32, shuffle=False)
    
            # Define the GAT model class
            class GAT(nn.Module):
                def __init__(self, hidden_channels, heads):
                    super(GAT, self).__init__()
    
                    # Define GAT convolution layers
                    self.conv1 = GATv2Conv(22, hidden_channels, heads=heads, concat=True)
                    self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
                    self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
    
                    # Define GraphNorm layers
                    self.gn1 = GraphNorm(hidden_channels * heads)
                    self.gn2 = GraphNorm(hidden_channels * heads)
                    self.gn3 = GraphNorm(hidden_channels * heads)
    
                    # Define the final linear layer
                    self.lin = nn.Linear(hidden_channels * heads, 2)
    
                def forward(self, x, edge_index, batch):
                    # Apply first GAT layer and normalization
                    x = self.conv1(x, edge_index)
                    x = F.relu(x)
                    x = self.gn1(x, batch)  # Apply GraphNorm
    
                    # Apply second GAT layer and normalization
                    x = self.conv2(x, edge_index)
                    x = F.relu(x)
                    x = self.gn2(x, batch)  # Apply GraphNorm
    
                    # Apply third GAT layer and normalization
                    x = self.conv3(x, edge_index)
                    x = self.gn3(x, batch)  # Apply GraphNorm
    
                    # Global pooling
                    x = global_mean_pool(x, batch)
    
                    # Apply dropout and final classifier
                    x = F.dropout(x, p=0.50, training=self.training)
                    x = self.lin(x)
    
                    return x
    
            # Initialize the model, optimizer, and loss function
            model = GAT(hidden_channels=22, heads=1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
    
            # Define the training function
            def train():
                model.train()
                for data in train_loader:
                    out = model(data.x, data.edge_index, data.batch)
                    loss = criterion(out, data.y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
    
            # Define the testing function
            def test(loader):
                model.eval()
                correct = 0
                for data in loader:
                    out = model(data.x, data.edge_index, data.batch)
                    pred = out.argmax(dim=1)
                    correct += int((pred == data.y).sum())
                return correct / len(loader.dataset)
            
            # Track the best accuracy and all accuracies for the current fold
            optimal_fold_acc = 0
            fold_test_accuracies = []
    
            # Train and test over multiple epochs
            for epoch in range(1, 500):
                train()
                train_acc = test(train_loader)
                test_acc = test(test_loader)
    
                # Store the test accuracy for this epoch
                fold_test_accuracies.append(test_acc)
    
                # Check if this epoch has the highest test accuracy so far
                if test_acc > optimal_fold_acc:
                    optimal_fold_acc = test_acc
    
            # Generate the name for the fold (e.g., '1v2', '1v3', etc.)
            fold_name = f"{train_fold_idx + 1}v{test_fold_idx + 1}"

            # Save fold accuracies (optimal, mean, high, low) for the current fold
            fold_accuracies.append({
                'fold_name': fold_name,
                'optimal': optimal_fold_acc,
                'mean': np.mean(fold_test_accuracies),
                'high': np.max(fold_test_accuracies),
                'low': np.min(fold_test_accuracies)
                })
        
            print(f'S{subject_number}: Optimal: {optimal_fold_acc}')
    
            # After the loop, store the accuracies for the current subject
            all_subjects_accuracies[f'S{subject_number}'] = fold_accuracies

# Optionally, save the accuracies dictionary to a file
with open('cross_session_accuracies.pkl', 'wb') as f:
    pickle.dump(all_subjects_accuracies, f)

# Example of printing the results
for subject, accuracies in all_subjects_accuracies.items():
    print(f'Subject {subject}:')
    for acc in accuracies:
        print(f'  Fold {acc["fold_name"]}: Optimal: {acc["optimal"]:.4f}, Mean: {acc["mean"]:.4f}, High: {acc["high"]:.4f}, Low: {acc["low"]:.4f}')
    

#% Plots

import pandas as pd

# Prepare data
data = []
for subject, folds in all_subjects_accuracies.items():
    for fold_data in folds:
        train_test_pair = fold_data['fold_name']
        accuracy = fold_data['optimal']  # Using 'optimal' accuracy here, you can change it to 'mean', 'high', or 'low'
        data.append({
            'Subject': subject,  # E.g., 'S1', 'S2', etc.
            'Train-Test Pair': train_test_pair,
            'Accuracy': accuracy
        })

df = pd.DataFrame(data)


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))
sns.boxplot(x='Subject', y='Accuracy', data=df)
plt.title('Accuracy Distribution Across Subjects')
plt.xlabel('Subject')
plt.ylabel('Accuracy')
plt.xticks(rotation=90)  # Rotate x-axis labels if needed
plt.tight_layout()
plt.show()

# Prepare data for heatmap
heatmap_data = df.pivot(index='Subject', columns='Train-Test Pair', values='Accuracy')

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title('Accuracy Heatmap Across Subjects and Train-Test Pairs')
plt.xlabel('Train-Test Pair')
plt.ylabel('Subject')
plt.xticks(ticks=range(len(heatmap_data.columns)), labels=heatmap_data.columns, rotation=90)
plt.yticks(ticks=range(len(heatmap_data.index)), labels=heatmap_data.index, rotation=0)
plt.tight_layout()
plt.show()


#% Average the Accuracies

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

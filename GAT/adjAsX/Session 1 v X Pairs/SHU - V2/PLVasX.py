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

#%%%%%%%%%%%% SHU Dataset - PLV as Feature %%%%%%%%%%%%%%%%

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


import time
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

# Dictionary to hold the loaded data for each subject
all_subjects_accuracies = {}

# Define the valid train-test pairs (1v2, 1v3, 1v4, 1v5)
valid_train_test_pairs = ['1v2', '1v3', '1v4', '1v5']

# Loop through each subject's data
for subject_number in subject_plv_data:
    all_session_data = subject_plv_data[subject_number]  # Data for all sessions of the subject
    
    # Initialize a list to store accuracy results for each permutation of train-test pairs
    subject_accuracies = []
    
    # Loop through each session to set it as the training set
    for train_session_idx in range(len(all_session_data)):
        traindata = all_session_data[train_session_idx]
        
        traind = []
        for i in range(np.size(traindata['plv_matrices'], 2)):
            traind.append(Data(x=traindata['adj_matrices'][:, :, i], edge_index=traindata['edge_indices'][:, :, i], y=traindata['labels'][i]))
        
        # Loop through each session again to set it as the test set, skipping the current train session
        for test_session_idx in range(len(all_session_data)):
            if test_session_idx == train_session_idx:
                continue  # Skip if the test session is the same as the train session

            # Only process the valid train-test pairs: 1v2, 1v3, 1v4, 1v5
            pair_name = f"{train_session_idx+1}v{test_session_idx+1}"
            if pair_name not in valid_train_test_pairs:
                continue  # Skip invalid pairs

            testdata = all_session_data[test_session_idx]
            
            testd = []
            for i in range(np.size(testdata['plv_matrices'], 2)):
                testd.append(Data(x=testdata['adj_matrices'][:, :, i], edge_index=testdata['edge_indices'][:, :, i], y=testdata['labels'][i]))
             
            # Define DataLoader for train and test sets
            train_loader = DataLoader(traind, batch_size=32, shuffle=False)
            test_loader = DataLoader(testd, batch_size=32, shuffle=False)
            
            # Define the GAT model class
            class GAT(nn.Module):
                def __init__(self, hidden_channels, heads):
                    super(GAT, self).__init__()

                    # Define GAT convolution layers
                    self.conv1 = GATv2Conv(32, hidden_channels, heads=heads, concat=True)
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
            
            # Define GAT model and other settings here
            model = GAT(hidden_channels=32, heads=1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Training and testing functions
            def train():
                model.train()
                for data in train_loader:
                    out = model(data.x, data.edge_index, data.batch)
                    loss = criterion(out, data.y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            def test(loader):
                model.eval()
                correct = 0
                start_time = time.time()  # Start time for inference
                for data in loader:
                    out = model(data.x, data.edge_index, data.batch)
                    pred = out.argmax(dim=1)
                    correct += int((pred == data.y).sum())
                end_time = time.time()  # End time for inference
                avg_inference_time = (end_time - start_time) / len(loader.dataset)  # Average inference time
                return correct / len(loader.dataset), avg_inference_time
            
            # Track accuracies and times for this train-test pair
            fold_test_accuracies = []
            inference_times = []
            optimal_accuracy = 0
            total_train_time = 0

            # Train and test over multiple epochs
            for epoch in range(1, 501):
                start_train_time = time.time()  # Start time for training
                train()
                end_train_time = time.time()  # End time for training
                total_train_time += (end_train_time - start_train_time)
                
                test_acc, avg_inference_time = test(test_loader)
                
                # Track accuracy and inference time for each epoch
                fold_test_accuracies.append(test_acc)
                inference_times.append(avg_inference_time)
                
                # Update optimal accuracy if the current epoch's test accuracy is the highest
                if test_acc > optimal_accuracy:
                    optimal_accuracy = test_acc
            
            # Store results for this train-test pair
            subject_accuracies.append({
                'pair': pair_name,
                'optimal': optimal_accuracy,
                'mean': np.mean(fold_test_accuracies),
                'high': np.max(fold_test_accuracies),
                'low': np.min(fold_test_accuracies),
                'train_time': total_train_time,
                'avg_inference_time': np.mean(inference_times)
            })
            
            print(f"Subject {subject_number}, Pair {pair_name}: Optimal Accuracy: {optimal_accuracy}, Train Time: {total_train_time:.4f}, Avg Inference Time: {np.mean(inference_times):.4f}")

    # After processing all pairs, save the accuracies for the subject
    all_subjects_accuracies[f'S{subject_number}'] = subject_accuracies

# Optionally, save the results to a file
with open('all_subjects_accuracies.pkl', 'wb') as f:
    pickle.dump(all_subjects_accuracies, f)

# Example: Print out all results
for subject, accuracies in all_subjects_accuracies.items():
    print(f'Subject {subject}:')
    for acc in accuracies:
        print(f'  Train-Test Pair {acc["pair"]}: Optimal: {acc["optimal"]:.4f}, Mean: {acc["mean"]:.4f}, High: {acc["high"]:.4f}, Low: {acc["low"]:.4f}')

# Prepare data for DataFrame (only for Fold 1v2, 1v3, 1v4, 1v5)
data = []
for subject, folds in all_subjects_accuracies.items():
    for fold_data in folds:
        train_test_pair = fold_data['pair']  # Use 'pair' as the fold name
        if train_test_pair in valid_train_test_pairs:  # Only include these folds
            accuracy = fold_data['optimal']  # 'optimal' accuracy is being used here
            data.append({
                'Subject': subject,  # E.g., 'S1', 'S2', etc.
                'Train-Test Pair': train_test_pair,
                'Accuracy': accuracy
            })

df = pd.DataFrame(data)

# Boxplot for accuracy distribution
plt.figure(figsize=(14, 8))
sns.boxplot(x='Subject', y='Accuracy', data=df)
plt.title("Accuracy Distribution for Train-Test Pairs 1v2, 1v3, 1v4, 1v5")
plt.xticks(rotation=90)  # Rotate subject names
plt.show()

# Remove the "S" prefix from Subject names
df['Subject'] = df['Subject'].str.replace('Ssub-', 'sub-', regex=False)

# Pivot data for heatmap
# Ensure subjects are on the y-axis and folds on the x-axis
heatmap_data = df.pivot(index="Subject", columns="Train-Test Pair", values="Accuracy")

# Heatmap with subjects on the y-axis and folds on the x-axis
plt.figure(figsize=(14, 8))

sns.heatmap(
    heatmap_data,  # Data for the heatmap
    annot=True,  # Display accuracy values
    cmap="YlGnBu",  # Color scheme similar to the uploaded image
    cbar_kws={'label': 'Accuracy'},  # Add colorbar label
    fmt=".2f",  # Format values with two decimal points
    linewidths=0.5,  # Add thin white lines between cells
    linecolor='white'  # White grid lines
)

# Set title and axis labels
plt.title("Accuracy Heatmap Across Subjects and Train-Test Pairs", fontsize=18, weight='bold', pad=20)
plt.xlabel("Train-Test Pair", fontsize=14, labelpad=15)
plt.ylabel("Subject", fontsize=14, labelpad=15)

# Rotate axis labels
plt.xticks(rotation=90, fontsize=12, weight='bold')  # Fold labels rotated vertically
plt.yticks(rotation=0, fontsize=12, weight='bold')   # Subject labels horizontal

# Adjust layout for better fit
plt.tight_layout()

# Show the heatmap
plt.show()


import numpy as np

# Assuming the dictionary is like the example above
for subject, folds in all_subjects_accuracies.items():
    print(f"Subject: {subject}")
    
    accuracies = []
    train_times = []
    inference_times = []
    
    # Now iterating directly over the list of fold dictionaries
    for details in folds:
        pair_name = details['pair']
        accuracy = details['optimal']  # Assuming 'optimal' is the accuracy
        train_time = details['train_time']
        inference_time = details['avg_inference_time']
        
        # Print the accuracies for each fold pair per subject
        print(f"  {pair_name} Accuracy: {accuracy:.4f}")
        
        accuracies.append(accuracy)
        train_times.append(train_time)
        inference_times.append(inference_time)
    
    # Calculate and print the mean train time and mean inference time for the subject
    mean_train_time = np.mean(train_times)
    mean_inference_time = np.mean(inference_times)
    
    print(f"  Mean Train Time for {subject}: {mean_train_time:.4f} seconds")
    print(f"  Mean Inference Time for {subject}: {mean_inference_time:.4f} seconds")
    print()  # Add a blank line between subjects

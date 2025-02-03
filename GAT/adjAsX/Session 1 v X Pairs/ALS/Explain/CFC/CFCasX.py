# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:26:26 2024
Rishan Patel, UCL, Bioelectronics Group.



https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
"""
import pandas as pd
import seaborn as sns
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
from scipy.integrate import simpson
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
import psutil
from torch_geometric.seed import seed_everything
from scipy.signal import butter, filtfilt, hilbert

# % Functions

def bandpasscfc(data, freq_range, sample_rate=256, order=5):
    nyquist = 0.5 * sample_rate
    low = freq_range[0] / nyquist
    high = freq_range[1] / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def cfcfcn(eegData, fs=256):
    # Use only the first 19 electrodes
    eegData = eegData[:, :19]
    
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
    
    # Use only the first 19 electrodes for CFC computation
    numElectrodes = 19  # Limit to the first 19 electrodes
    numTrials = subject_data['L'].shape[1]
    cfc = {field: np.zeros((numElectrodes, numElectrodes, numTrials)) for field in idx}

    for i, field in enumerate(idx):
        for j in range(numTrials):
            x = subject_data[field][0, j]  # Extract EEG data for the field (L or R)
            cfc[field][:, :, j] = cfcfcn(x, fs=fs)

    l, r = cfc['L'], cfc['R']
    yl, yr = np.zeros((numTrials, 1)), np.ones((numTrials, 1))
    
    # Concatenate the matrices from the left and right electrodes
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



def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = sig.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sig.sosfiltfilt(sos, data)
    return filtered_data


def get_model_memory(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.element_size() * p.numel() for p in model.parameters()) + \
        sum(b.element_size() * b.numel() for b in model.buffers())
    return total_params, total_size / (1024 ** 2)  # Return memory in MB


def get_ram_usage():
    process = psutil.Process()  # Current process
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # Return RAM usage in GB


# %

# Set the random seed for reproducibility
seed_everything(12345) # Seed for everything
# % Preparing Data
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data'

# Define the subject numbers
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]

# Dictionary to hold the loaded data for each subject
subject_data = {}
all_subjects_accuracies = {}

# Loop through the subject numbers and load the corresponding data
for subject_number in subject_numbers:
    mat_fname = pjoin(data_dir, f'S{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_data[f'S{subject_number}'] = mat_contents[f'Subject{subject_number}']
    S1 = subject_data[f'S{subject_number}'][:, :-1]
    
    plv, y = compute_cfc(S1)
    threshold = 0.1
    graphs = create_graphs(plv, threshold)
    numElectrodes = 19
    adj = np.zeros([numElectrodes, numElectrodes, len(graphs)])
    for i, G in enumerate(graphs):
        adj[:, :, i] = nx.to_numpy_array(G)

    adj = torch.tensor(adj, dtype=torch.float32)
    # % Initialize an empty list to store edge indices
    edge_indices = []  # % Edge indices are a list of source and target nodes in a graph. Think of it like the adjacency matrix

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
        edge_index = torch.tensor(
            [source_nodes, target_nodes], dtype=torch.long)

        # Append edge index to the list
        edge_indices.append(edge_index)

    # Stack all edge indices along a new axis to create a 2D tensor
    edge_indices = torch.stack(edge_indices, dim=-1)

    del col, edge_index, i, row, source_nodes, target_nodes

    from torch_geometric.data import Data

    data_list = []
    for i in range(np.size(adj, 2)):
        data_list.append(
            Data(x=adj[:, :, i], edge_index=edge_indices[:, :, i], y=y[i, 0]))

    datal = []
    datar = []
    size = len(data_list)
    idx = size//2
    c = [0, idx, idx*2, idx*3]

    datal = data_list[c[0]:c[1]]
    datar = data_list[c[1]:c[2]]

    data_list = []

    for i in range(idx):
        x = [datal[i], datar[i]]  # datare[i]]
        data_list.extend(x)

    size = len(data_list)

    # Initialize a dictionary to store accuracies for each subject

    # Assuming 'data_list' is your data for the current subject
    num_folds = 4
    fold_size = len(data_list) // num_folds

    # Split the data manually into 4 folds
    folds = [data_list[i * fold_size:(i + 1) * fold_size]
             for i in range(num_folds)]

    fold_accuracies = []
  # Iterate over each fold as training data
    for train_fold_idx in range(num_folds):
        # Only process folds where train_fold_idx is 0 (fold 1)
        if train_fold_idx != 0:
            continue  # Skip other folds as training data

        # Iterate over each fold as testing data
        for test_fold_idx in range(num_folds):
            # Exclude folds not part of the 1vX comparisons
            if test_fold_idx == train_fold_idx or test_fold_idx == 0:
                continue  # Skip if testing fold is the same as training fold or fold 1

            # Create training and testing sets
            train = folds[train_fold_idx]
            test = folds[test_fold_idx]

            

            # Create DataLoader for train and test sets
            train_loader = DataLoader(train, batch_size=32, shuffle=False)
            test_loader = DataLoader(test, batch_size=32, shuffle=False)

            # Define the GAT model class

            class GAT(nn.Module):
                def __init__(self, hidden_channels, heads):
                    super(GAT, self).__init__()

                    # Define GAT convolution layers
                    self.conv1 = GATv2Conv(
                        19, hidden_channels, heads=heads, concat=True)
                    self.conv2 = GATv2Conv(
                        hidden_channels * heads, hidden_channels, heads=heads, concat=True)
                    self.conv3 = GATv2Conv(
                        hidden_channels * heads, hidden_channels, heads=heads, concat=True)

                    # Define GraphNorm layers
                    self.gn1 = GraphNorm(hidden_channels * heads)
                    self.gn2 = GraphNorm(hidden_channels * heads)
                    self.gn3 = GraphNorm(hidden_channels * heads)

                    # Define the final linear layer
                    self.lin = nn.Linear(hidden_channels * heads, 2)

                def forward(self, x, edge_index, batch, return_attention_weights=True):
                    # print(f"Batch size: {batch.size()}")  # Check the batch tensor

                    # Apply first GAT layer and normalization
                    w1, att1 = self.conv1(
                        x, edge_index, return_attention_weights=True)
                    w1 = F.relu(w1)
                    w1 = self.gn1(w1, batch)  # Apply GraphNorm

                    # Apply second GAT layer and normalization
                    w2, att2 = self.conv2(
                        w1, edge_index, return_attention_weights=True)
                    w2 = F.relu(w2)
                    w2 = self.gn2(w2, batch)  # Apply GraphNorm

                    # Apply third GAT layer and normalization
                    w3, att3 = self.conv3(
                        w2, edge_index, return_attention_weights=True)
                    w3 = self.gn3(w3, batch)  # Apply GraphNorm

                    # print(f"Shape of w3 before pooling: {w3.size()}")  # Before global mean pooling

                    # Global pooling
                    w3 = global_mean_pool(w3, batch)
                    # print(f"Shape of w3 after pooling: {w3.size()}")  # After global mean pooling

                    # Apply dropout and final classifier
                    w3 = F.dropout(w3, p=0.50, training=self.training)
                    o = self.lin(w3)

                    return o, w3, att3

            # Initialize the model, optimizer, and loss function
            model = GAT(hidden_channels=19, heads=1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            # Define the training function
            def train():
                model.train()
                start_time = time.time()  # Start timing
                all_edge = []
                all_att = []  # List to accumulate attention scores for all batches
                all_w = []
                for data in train_loader:
                    out, w3, att3 = model(data.x, data.edge_index, data.batch)

                    # Append the attention scores for the current batch to the list
                    all_att.append(att3[1])
                    all_edge.append(att3[0])
                    all_w.append(w3)
                    
                    data.y = data.y.long()  # Ensure the labels are of type Long
                    loss = criterion(out, data.y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    ## Monitor memory usage
                    #ram_usage = get_ram_usage()
                    #print(f"Batch RAM Usage: {ram_usage:.2f} GB")

                end_time = time.time()  # End timing
                # Concatenate attention scores from all batches
                all_att = torch.cat(all_att, dim=0)
                all_edge = torch.cat(all_edge, dim=1)
                all_w = torch.cat(all_w, dim=0)
                
                # Return accumulated attention scores
                return end_time - start_time, all_w, all_att, all_edge

            def test(loader):
                model.eval()
                correct = 0
                inference_start = time.time()  # Start timing
                all_edge = []
                all_w = []
                all_att = []  # List to accumulate attention scores for all batches
                for data in loader:
                    out, w3, att3 = model(data.x, data.edge_index, data.batch)

                    # Append the attention scores for the current batch to the list
                    all_att.append(att3[1])
                    all_edge.append(att3[0])
                    all_w.append(w3)

                    pred = out.argmax(dim=1)
                    #print(pred.numpy().size)
                    correct += int((pred == data.y).sum())
                    #print(correct)

                    # Monitor memory usage
                    # ram_usage = get_ram_usage()
                    # print(f"Batch RAM Usage: {ram_usage:.2f} GB")

                inference_end = time.time()  # End timing
                inference_time = inference_end - inference_start  # Calculate inference time

                # Concatenate attention scores from all batches
                all_att = torch.cat(all_att, dim=0)
                all_edge = torch.cat(all_edge, dim=1)
                all_w = torch.cat(all_w, dim=0)

                # Return accumulated attention scores
                return correct / len(loader.dataset), inference_time, all_w, all_att, all_edge

            # Track the best accuracy, training times, and testing times
            optimal_fold_acc = 0  # Store only the accuracy (float)
            fold_test_accuracies = []
            fold_weights = {}
            fold_att = {}
            train_times = []
            inference_times = []
            ram = []

            start_train_time = time.time()  # Start timing for the entire training process

            # Train and test over multiple epochs
            for epoch in range(1, 501):  # Note: Range is 1 to 500 (inclusive)
                # Record training time for each epoch
                epoch_train_time, trainw3, trainatt3,trainedge3 = train()
                
                train_times.append(epoch_train_time)

                # Get train and test accuracy, along with inference time
                # train_acc, _ = test(train_loader)
                test_acc, inference_time, testw3, testatt3, testedge3 = test(test_loader)
                inference_times.append(inference_time)
                # Monitor memory usage
                ram_usage = get_ram_usage()

                # Store the test accuracy for this epoch
                fold_test_accuracies.append(test_acc)

                # Check if this epoch has the highest test accuracy so far
                # Compare accuracy only (not tuple)
                if test_acc > optimal_fold_acc:
                    optimal_fold_acc = test_acc
                    # print(f"{epoch}: {optimal_fold_acc}")

                    # Store model weights and attention weights
                    traindetails = {
                        'att3': trainatt3.detach().cpu().numpy(),
                        'edge3': trainedge3.detach().cpu().numpy(),
                        'w3': trainw3.detach().cpu().numpy(),
                    }
                    testdetails = {
                        'att3': testatt3.detach().cpu().numpy(),
                        'edge3': testedge3.detach().cpu().numpy(),
                        'w3': testw3.detach().cpu().numpy(),
                    }

                    # Save the model weights
                    torch.save(model.state_dict(
                    ), f"best_model_fold_{train_fold_idx + 1}v{test_fold_idx + 1}.pth")

            # End timing for the entire training process
            end_train_time = time.time()
            # Total time for training across all epochs
            total_train_time = end_train_time - start_train_time

            # Generate the name for the fold (e.g., '1v2', '1v3', etc.)
            fold_name = f"{train_fold_idx + 1}v{test_fold_idx + 1}"

            # Save fold accuracies, timings, and best weights/attention scores for the current fold
            fold_accuracies.append({
                'fold_name': fold_name,
                'optimal': optimal_fold_acc,
                'mean': np.mean(fold_test_accuracies),
                'high': np.max(fold_test_accuracies),
                'low': np.min(fold_test_accuracies),
                'total_train_time': total_train_time,
                'avg_train_time': np.mean(train_times),
                'avg_inference_time': np.mean(inference_times),
                'traindetails': traindetails,  # Add the best model weights
                'testdetails': testdetails,  # Add the best attention scores
                'ram': ram_usage
            })

            # Print results for the fold, including weights/attention data
            print(f'S{subject_number}, Fold {fold_name}: Optimal: {optimal_fold_acc:.4f}, '
                  f'Total Train Time: {total_train_time:.4f}s, '
                  f'Avg Train Time per Epoch: {np.mean(train_times):.4f}s, '
                  f'Avg Inference Time per Epoch: {np.mean(inference_times):.4f}s')

            # After the loop, store the accuracies for the current subject
            all_subjects_accuracies[f'S{subject_number}'] = fold_accuracies
    


total_params, model_memory = get_model_memory(model)
print(f"Model Total Parameters: {total_params}")
print(f"Model Memory: {model_memory:.2f} MB")

#%%

import pickle
import os

# Define the save directory and file name
save_dir = r"C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Understanding Stability in Graphical Representation\GAT\adjAsX\Session 1 v X Pairs\ALS\Explain\CFC"
file_name = "all_subjects_accuracies.pkl"
save_file_path = os.path.join(save_dir, file_name)

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Save the dictionary to a file
try:
    with open(save_file_path, 'wb') as f:
        pickle.dump(all_subjects_accuracies, f)
    print(f"Dictionary saved successfully at: {save_file_path}")
except Exception as e:
    print(f"Error saving dictionary: {e}")

#% Plots and Stats
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

#%%
import numpy as np

def process_subject_accuracies(data):
    # Subjects of interest
    subjects_of_interest = {'S1', 'S2', 'S5', 'S9', 'S21', 'S31', 'S34', 'S39'}
    optimal_percentages = {}
    ram_values = []
    av_inference_times = []
    total_train_times = []

    # Loop through all subjects
    for subject, train_test_pairs in data.items():
        # Check if the subject is in the list of interest
        if subject in subjects_of_interest:
            for pair_idx, pair_data in enumerate(train_test_pairs):
                # Extract and process the 'optimal' field
                optimal = pair_data.get('optimal')
                if optimal is not None:
                    # Convert optimal to percentage and round to 2 dp
                    key = f"{subject}_pair{pair_idx + 1}"
                    optimal_percentages[key] = round(float(optimal) * 100, 2)

        # Collect data for the running mean
        for pair_data in train_test_pairs:
            ram = pair_data.get('ram')
            av_inference_time = pair_data.get('avg_inference_time')
            total_train_time = pair_data.get('total_train_time')

            # Append to respective lists only if the value is not None
            if ram is not None:
                ram_values.append(float(ram))
            if av_inference_time is not None:
                av_inference_times.append(float(av_inference_time))
            if total_train_time is not None:
                total_train_times.append(float(total_train_time))

    # Calculate running means, ensuring no division by zero
    running_mean = {
        'ram': np.mean(ram_values) if ram_values else 0.0,
        'av_inference_time': np.mean(av_inference_times) if av_inference_times else 0.0,
        'total_train_time': np.mean(total_train_times) if total_train_times else 0.0,
    }

    return optimal_percentages, running_mean

# Example usage with `all_subjects_accuracies`
optimal_percentages, running_mean = process_subject_accuracies(all_subjects_accuracies)

# Print results
print("Optimal Percentages for Specific Subjects and Train-Test Pairs:")
for key, value in optimal_percentages.items():
    print(f"{key}: {value}%")

print("\nRunning Mean for the Whole Cohort:")
for key, value in running_mean.items():
    print(f"{key}: {value:.4f}")

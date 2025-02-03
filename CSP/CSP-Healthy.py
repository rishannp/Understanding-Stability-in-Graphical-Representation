"""
Created on Tuesday 30th July 2024 at 12:22pm
Rishan Patel, Bioelectronics and Aspire Create Group, UCL


https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
"""

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
from scipy.stats import zscore


#%%%%%%%%%%%% SHU Dataset - CSP %%%%%%%%%%%%%%%%

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

def zscore_normalize_trials(filtered_data_split):
    """Apply Z-score normalization to each channel of each trial."""
    normalized_data_split = {}

    for subject in filtered_data_split:
        subject_data = filtered_data_split[subject]
        normalized_subject_data = {}

        for direction in ['L', 'R']:
            trials = subject_data[direction]  # Shape: (trials, channels, samples)
            normalized_trials = []

            for trial in range(trials.shape[0]):
                trial_data = trials[trial, :, :]  # Shape: (channels, samples)

                # Apply Z-score normalization to each channel of the trial
                normalized_trial_data = np.zeros_like(trial_data)
                for ch in range(trial_data.shape[0]):
                    normalized_trial_data[ch, :] = zscore(trial_data[ch, :], axis=-1)  # Normalize each channel across time

                normalized_trials.append(normalized_trial_data)

            # Convert the list of normalized trials back to a NumPy array
            normalized_subject_data[direction] = np.array(normalized_trials)

        normalized_data_split[subject] = normalized_subject_data

    return normalized_data_split

# Apply Z-score normalization to the filtered data
normalized_data_split = zscore_normalize_trials(filtered_data_split)

# Optionally, delete the original filtered data to free up memory
del filtered_data_split


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
fold_data = prepare_folds_for_subjects(normalized_data_split)

# Example of accessing the folds
for subject, data in fold_data.items():
    for direction, folds in data.items():
        for i, fold in enumerate(folds):
            print(f"Subject: {subject}, Direction: {direction}, Fold {i+1} shape: {fold.shape}")

#%%

def train_and_test_csp(train_data, test_data, n_components=10, plot_csp=True):
    """Train CSP on train_data and test on test_data."""
    # Initialize CSP with normalization by trace and mutual information
    csp = CSP(n_components=n_components, log=True, norm_trace=True, component_order='mutual_info')
    
    # Prepare data
    X_train, y_train = train_data['data'].astype(np.float64), train_data['labels']
    X_test, y_test = test_data['data'].astype(np.float64), test_data['labels']
    
    # Fit CSP on training data
    csp.fit(X_train, y_train)
    
    # Plot CSP patterns if requested
    if plot_csp:
        # Prepare the channel names and their respective 3D coordinates
        ch_names = ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 
                    'CZ', 'C3', 'C4', 'T3', 'T4', 'A1', 'A2', 'CP1', 'CP2', 'CP5', 'CP6', 
                    'PZ', 'P3', 'P4', 'T5', 'T6', 'PO3', 'PO4', 'OZ', 'O1', 'O2']

        coords = [
            [80.79573128, 26.09631015, -4.00404831],  # FP1
            [80.79573128, -26.09631015, -4.00404831], # FP2
            [60.73017777, 0, 59.47138394],            # FZ
            [57.57616305, 48.14114469, 39.90508284],  # F3
            [57.57616305, -48.14114469, 39.90508284], # F4
            [49.88651615, 68.41148946, -7.49690713],  # F7
            [49.88728633, -68.41254564, -7.482129533],# F8
            [32.43878889, 32.32575332, 71.60845375],  # FC1
            [32.43878889, -32.32575332, 71.60845375], # FC2
            [28.80808576, 76.2383868, 24.1413043],    # FC5
            [28.80808576, -76.2383868, 24.1413043],   # FC6
            [5.20E-15, 0, 85],                        # CZ
            [3.87E-15, 63.16731017, 56.87610154],     # C3
            [3.87E-15, -63.16731017, 56.87610154],    # C4
            [5.17e-15, 84.5, -8.85],                  # T3
            [5.17e-15, -84.5, -8.85],                 # T4
            [3.68e-15, 60.1, -60.1],                  # A1
            [3.68e-15, -60.1, -60.1],                 # A2
            [-32.38232042, 32.38232042, 71.60845375], # CP1
            [-32.38232042, -32.38232042, 71.60845375],# CP2
            [-29.2068723, 76.08650365, 24.1413043],   # CP5
            [-29.2068723, -76.08650365, 24.1413043],  # CP6
            [-60.73017777, -7.44E-15, 59.47138394],   # PZ
            [-57.49205325, 48.24156068, 39.90508284], # P3
            [-57.49205325, -48.24156068, 39.90508284],# P4
            [-49.9, 68.4, -7.49],                     # T5
            [-49.9, -68.4, -7.49],                    # T6
            [-76.40259649, 30.8686527, 20.8511278],   # PO3
            [-76.40259649, -30.8686527, 20.8511278],  # PO4
            [-84.9813581, -1.04E-14, -1.78010569],    # OZ
            [-80.75006159, 26.23728548, -4.00404831], # O1
            [-80.75006159, -26.23728548, -4.00404831] # O2
        ]
        
        # Create a dictionary for the montage
        ch_pos = {ch: coord for ch, coord in zip(ch_names, coords)}
        
        # Create the montage
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
        
        # Create the Info object
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')  # Adjust 'sfreq' as needed
        info.set_montage(montage)  # Attach the montage to the Info object

        # Plot the topographic patterns of the CSP components
        csp.plot_patterns(info, ch_type='eeg', components=list(range(n_components)), show=True)
    
    # Transform both train and test data
    X_train_transformed = csp.transform(X_train)
    X_test_transformed = csp.transform(X_test)
    
    # Standardize data
    scaler = StandardScaler()
    X_train_transformed = scaler.fit_transform(X_train_transformed)
    X_test_transformed = scaler.transform(X_test_transformed)
    
    # Train a classifier (SVM)
    classifier = SVC()
    classifier.fit(X_train_transformed, y_train)
    
    # Test the classifier
    y_pred = classifier.predict(X_test_transformed)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def prepare_data_for_csp(folds_L, folds_R):
    """Prepare data and labels for CSP with concatenated Left and Right data."""
    fold_data = {}
    for i in range(len(folds_L)):
        # Concatenate Left and Right data
        data = np.concatenate((folds_L[i], folds_R[i]), axis=0)
        
        # Generate labels: 1 for Left, 2 for Right
        labels_L = np.ones(folds_L[i].shape[0])
        labels_R = np.full(folds_R[i].shape[0], 2)
        labels = np.concatenate((labels_L, labels_R), axis=0)
        
        fold_data[f'fold_{i+1}'] = {
            'data': data,
            'labels': labels
        }
    
    return fold_data

def split_into_folds(data, n_folds=5):
    """Split data into n_folds, evenly split."""
    n_trials = data.shape[0]
    fold_size = n_trials // n_folds
    
    folds = []
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = None if i == n_folds - 1 else (i + 1) * fold_size
        folds.append(data[start_idx:end_idx])
    
    return folds


# Initialize a dictionary to store accuracies
accuracies = {}

# Iterate over each subject
for subject, data in normalized_data_split.items():
    # Split the data into folds for Left and Right
    folds_L = split_into_folds(data['L'])
    folds_R = split_into_folds(data['R'])
    
    # Prepare data for CSP
    prepared_folds = prepare_data_for_csp(folds_L, folds_R)
    
    # Iterate over all pairs of folds
    fold_names = list(prepared_folds.keys())
    for i in range(len(fold_names)):
        for j in range(len(fold_names)):
            if i != j:  # Ensure that training and testing folds are different
                train_fold_key = fold_names[i]
                test_fold_key = fold_names[j]
                
                train_fold = prepared_folds[train_fold_key]
                test_fold = prepared_folds[test_fold_key]
                
                # Train and test CSP
                accuracy = train_and_test_csp(train_fold, test_fold)
                
                # Store accuracy in dictionary
                fold_pair = f'{train_fold_key}_vs_{test_fold_key}'
                if subject not in accuracies:
                    accuracies[subject] = {}
                accuracies[subject][fold_pair] = accuracy
                
                print(f'Subject: {subject}, Training Fold: {train_fold_key}, Testing Fold: {test_fold_key}, Accuracy: {accuracy}')
                
#%% Plots

import pandas as pd

# Prepare data
data = []
for subject, pairs in accuracies.items():
    for pair, accuracy in pairs.items():
        train_sess, test_sess = pair.split('_vs_')
        data.append({
            'Subject': f'{subject}',  # Use 'S1', 'S2', etc. for subjects
            'Train-Test Pair': f'{train_sess} vs {test_sess}',
            'Accuracy': accuracy
        })

df = pd.DataFrame(data)

import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot
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

#%%%%%%%%%%%%%%%% GAT  %%%%%%%%%%%%%%%%%

import json

# Load the JSON data from the file
with open("SHU_CSP_accuracies.json", "r") as file:
    accuracies_data = json.load(file)

# Function to calculate the average accuracy for each subject
def calculate_average_accuracies(data):
    average_accuracies = {}

    for subject, folds in data.items():
        total_accuracy = 0
        count = 0
        
        for accuracy in folds.values():
            total_accuracy += accuracy
            count += 1
        
        average_accuracy = total_accuracy / count
        average_accuracies[subject] = average_accuracy
    
    return average_accuracies

# Calculate the average accuracies
average_accuracies = calculate_average_accuracies(accuracies_data)

# Print the results
for subject, avg_accuracy in average_accuracies.items():
    print(f"{subject}: {avg_accuracy:.4f}")

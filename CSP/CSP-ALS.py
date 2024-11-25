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

#%%%%%%%%%%%% ALS Dataset - CSP %%%%%%%%%%%%%%%%

# Define the channels
channels = np.array([
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 
    'T7', 'C3', 'CZ', 'C4', 'T8', 
    'P7', 'P3', 'PZ', 'P4', 'P8', 
    'O1', 'O2'
])

# Define the directory containing your .mat files
directory = r'C:\Users\uceerjp\Desktop\PhD\Penn State Data\Work\Data\OG_Full_Data'

# Create a dictionary to store the concatenated data and labels by subject
data_by_subject = {}

# Define the list of subject IDs you're interested in
subject_ids = [1, 2, 5, 9, 21, 31, 34, 39]

# Function to remove the last entry in the list if it's all zeros
def remove_last_entry_if_all_zeros(data_list):
    if len(data_list) > 0:
        if np.all(data_list[-1] == 0):
            return data_list[:-1]
    return data_list

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a .mat file
    if filename.endswith('.mat'):
        # Extract the numeric part from the filename assuming format 'OGFS<number>.mat'
        subject_number_str = filename[len('OGFS'):-len('.mat')]
        
        # Check if the subject number is one you're interested in
        if subject_number_str.isdigit() and int(subject_number_str) in subject_ids:
            # Generate the full path to the file
            file_path = os.path.join(directory, filename)
            
            # Load the .mat file
            mat_data = loadmat(file_path)
            
            # Extract the variable name (usually 'SubjectX')
            subject_variable_name = f'Subject{subject_number_str}'
            
            # Check if the expected variable exists in the .mat file
            if subject_variable_name in mat_data:
                # Access the 1xN array of void192
                void_array = mat_data[subject_variable_name]
                
                # Initialize the subject entry in the dictionary if not already present
                subject_id = f'S{subject_number_str}'
                if subject_id not in data_by_subject:
                    data_by_subject[subject_id] = {
                        'L': [],
                        'R': [],
                    }
                
                # Loop through each element in the void array
                for item in void_array[0]:
                    # Extract the 'L', 'R', and 'Re' fields
                    L_data = item['L']
                    R_data = item['R']
                    
                    # Append data to the respective lists
                    data_by_subject[subject_id]['L'].append(L_data)
                    data_by_subject[subject_id]['R'].append(R_data)
                
                # Clean up the lists by removing the last entry if it is full of zeros
                data_by_subject[subject_id]['L'] = remove_last_entry_if_all_zeros(data_by_subject[subject_id]['L'])
                data_by_subject[subject_id]['R'] = remove_last_entry_if_all_zeros(data_by_subject[subject_id]['R'])

# Clean up unnecessary variables
del directory, file_path, filename, item, mat_data, subject_id, subject_ids, subject_number_str, subject_variable_name, void_array

#%%

fs= 256

# Bandpass filter function
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

        # Apply filter to each direction: 'L', 'R', and 'Re'
        for direction in ['L', 'R']:
            trials = subject_data[direction]  # List of trials, each trial is an array (Samples, 22 Channels)
            filtered_trials = []

            for trial_data in trials:
                # Remove the last 3 channels, reducing it to (Samples, 19 Channels)
                trial_data = trial_data[:, :19]

                # Apply bandpass filter to each channel
                filtered_trial_data = np.zeros_like(trial_data)
                for ch in range(trial_data.shape[1]):  # Loop over channels (now 19 channels)
                    filtered_trial_data[:, ch] = filtfilt(b, a, trial_data[:, ch])

                filtered_trials.append(filtered_trial_data)

            # Keep the list of filtered trials as a list instead of converting to a NumPy array
            filtered_subject_data[direction] = filtered_trials

        filtered_data_split[subject] = filtered_subject_data

    return filtered_data_split

# Apply bandpass filter to the data_split
filtered_data_split = bandpass_filter_trials(data_by_subject, low_freq=8, high_freq=30, sfreq=fs)

# Optionally delete the original data to free up memory
del data_by_subject
#%%

def split_into_folds(trials, n_splits=4):
    """Splits a list of trials into n_splits causal folds."""
    n_trials = len(trials)
    trials_per_fold = n_trials // n_splits
    
    folds = []
    
    for i in range(n_splits):
        start_index = i * trials_per_fold
        if i == n_splits - 1:
            end_index = n_trials  # Last fold gets all remaining trials
        else:
            end_index = (i + 1) * trials_per_fold
        
        # Append the list of trials for this fold
        folds.append(trials[start_index:end_index])
    
    return folds

def prepare_folds_for_subjects(filtered_data_split, n_splits=4):
    """Splits filtered data into causal folds for each subject and direction."""
    fold_data = {}

    for subject in filtered_data_split:
        subject_data = filtered_data_split[subject]
        fold_data[subject] = {}
        
        for direction in ['L', 'R']:
            trials = subject_data[direction]  # List of trials, each trial is a NumPy array
            folds = split_into_folds(trials, n_splits=n_splits)
            
            fold_data[subject][direction] = folds
    
    return fold_data

# Apply the fold splitting to filtered data
fold_data = prepare_folds_for_subjects(filtered_data_split)

# Example of accessing the folds
for subject, data in fold_data.items():
    for direction, folds in data.items():
        for i, fold in enumerate(folds):
            if fold:  # Check if fold is not empty
                print(f"Subject: {subject}, Direction: {direction}, Fold {i+1} has {len(fold)} trials, First trial shape: {fold[0].shape}")

del direction, filtered_data_split, fold, folds, i, subject,data


#%%
def merge_l_r_folds_with_labels(fold_data):
    """Merge L and R folds for each subject, adding labels for each trial."""
    merged_folds = {}
    
    for subject, data in fold_data.items():
        merged_folds[subject] = []
        
        num_folds = len(data['L'])
        
        for i in range(num_folds):
            # Get the L and R folds
            L_fold = data['L'][i]
            R_fold = data['R'][i]
            
            # Merge the lists directly
            combined_folds = L_fold + R_fold
            
            # Create labels: 1 for L trials, 2 for R trials
            labels_L = [1] * len(L_fold)
            labels_R = [2] * len(R_fold)
            combined_labels = labels_L + labels_R
            
            # Append the combined list and labels to the merged_folds dictionary
            merged_folds[subject].append({
                'data': combined_folds,
                'labels': combined_labels
            })
    
    return merged_folds

# Assuming fold_data is your dictionary with subjects, L, and R folds
merged_folds = merge_l_r_folds_with_labels(fold_data)

# Example: Accessing the merged folds for Subject 'S1'
for i, fold_info in enumerate(merged_folds['S1']):
    print(f"Fold {i+1} has {len(fold_info['data'])} trials and labels with length: {len(fold_info['labels'])}")

#%%

# Initialize maxlen to store maximum lengths for each subject and each fold
# Assuming there are 9 subjects and 4 folds per subject
maxlen = np.zeros((9, 4), dtype=int)

# Iterate over each subject
for subj_idx, subject in enumerate(merged_folds):
    # Iterate over each fold for the current subject
    for fold_idx, fold in enumerate(merged_folds[subject]):
        # Convert data to numpy array with dtype=object
        data = np.array(fold['data'], dtype=object)
        
        # Initialize max length for this fold
        max_length = 0
        
        # Iterate through each trial in the fold
        for trial in data:
            # Check if the trial is 2D
            if trial.ndim == 2:
                length = trial.shape[0]  # Sample dimension (number of columns)
            else:
                raise ValueError("Each trial should be a 2D array (trials x samples).")
            
            # Update the maximum length for this fold
            if length > max_length:
                max_length = length
        
        # Store the maximum length found for the current subject and fold
        maxlen[subj_idx, fold_idx] = max_length

# Print the maximum lengths
print("Maximum sample lengths for each subject and fold:")
print(maxlen)


# Initialize the padded data dictionary
padded_data = {}

# Iterate over each subject
for subj_idx, subject in enumerate(merged_folds):
    padded_data[subject] = []
    
    # Iterate over each fold for the current subject
    for fold_idx, fold in enumerate(merged_folds[subject]):
        # Get the maximum length for the current fold
        max_length = maxlen[subj_idx, fold_idx]
        
        # Convert data to numpy array with dtype=object
        data = np.array(fold['data'], dtype=object)
        
        # Initialize padded array
        num_trials = len(data)
        num_channels = data[0].shape[1]  # Number of channels (assuming all trials have the same number of channels)
        padded_fold_data = np.zeros((num_trials, num_channels, max_length), dtype=np.float64)
        
        # Fill the padded array with the data
        for trial_idx, trial in enumerate(data):
            num_samples = trial.shape[0]
            if num_samples > max_length:
                raise ValueError(f"Trial data length {num_samples} exceeds max length {max_length}")
            # Assuming trial.shape is (num_samples, num_channels)
            padded_fold_data[trial_idx, :, :num_samples] = trial.T  # Transpose to fit into (num_channels, max_length)
        
        # Append the padded fold data to the subject's data
        padded_data[subject].append({
            'data': padded_fold_data,
            'labels': fold['labels']  # Copy the labels as is
        })

# Example: Accessing the padded data for Subject 'S1'
for i, fold_info in enumerate(padded_data['S1']):
    print(f"Fold {i+1} has padded data with shape: {fold_info['data'].shape}")


#%%

def train_and_test_csp(train_data, test_data, n_components=10, plot_csp=True):
    """Train CSP on train_data and test on test_data."""
    # Initialize CSP with normalization by trace and mutual information
    csp = CSP(n_components=n_components, log=True, norm_trace=True, component_order='mutual_info')
    
    # Prepare data
    X_train = np.array(train_data['data'])  # Shape: [trials, channels, samples]
    y_train = np.array(train_data['labels'])  # Shape: [trials]
    X_test = np.array(test_data['data'])  # Shape: [trials, channels, samples]
    y_test = np.array(test_data['labels'])  # Shape: [trials]
    
    # Fit CSP on training data
    csp.fit(X_train, y_train)
    
    # Plot CSP patterns if requested
    if plot_csp:
        # Prepare the channel names and their respective coordinates
        ch_names = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']
        coords = [
            [0.950, 0.309, -0.0349],  # FP1
            [0.950, -0.309, -0.0349], # FP2
            [0.587, 0.809, -0.0349],  # F7
            [0.673, 0.545, 0.500],    # F3
            [0.719, 0, 0.695],        # FZ
            [0.673, -0.545, 0.500],   # F4
            [0.587, -0.809, -0.0349], # F8
            [6.120e-17, 0.999, -0.0349], # T7
            [4.400e-17, 0.719, 0.695],   # C3
            [3.750e-33, -6.120e-17, 1],   # CZ
            [4.400e-17, -0.719, 0.695],  # C4
            [6.120e-17, -0.999, -0.0349],# T8
            [-0.587, 0.809, -0.0349],    # P7
            [-0.673, 0.545, 0.500],      # P3
            [-0.719, -8.810e-17, 0.695], # PZ
            [-0.673, -0.545, 0.500],     # P4
            [-0.587, -0.809, -0.0349],   # P8
            [-0.950, 0.309, -0.0349],    # O1
            [-0.950, -0.309, -0.0349]    # O2
        ]
        
        # Create a dictionary for the montage
        ch_pos = {ch: coord for ch, coord in zip(ch_names, coords)}
        
        # Create the montage
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
        
        # Create the Info object
        info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')  # Adjust 'sfreq' as needed
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

# Initialize a dictionary to store accuracies
accuracies = {}

# Iterate over each subject in padded_data
for subject, folds in padded_data.items():
    num_folds = len(folds)
    
    # Iterate over all pairs of folds
    for i in range(num_folds):
        for j in range(num_folds):
            if i != j:  # Ensure that training and testing folds are different
                train_fold = folds[i]
                test_fold = folds[j]
                
                # Train and test CSP
                accuracy = train_and_test_csp(train_fold, test_fold)
                
                # Store accuracy in dictionary
                fold_pair = f'fold_{i+1}_vs_fold_{j+1}'
                if subject not in accuracies:
                    accuracies[subject] = {}
                accuracies[subject][fold_pair] = accuracy
                
                print(f'Subject: {subject}, Training Fold: fold_{i+1}, Testing Fold: fold_{j+1}, Accuracy: {accuracy}')


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

# Load the JSON data
with open('ALS_CSP_accuracies.json', 'r') as file:
    data = json.load(file)

# Initialize a dictionary to store the average accuracies
average_accuracies = {}

# Iterate over each subject in the dataset
for subject, accuracies in data.items():
    # Calculate the average accuracy for the subject
    avg_accuracy = sum(accuracies.values()) / len(accuracies)
    average_accuracies[subject] = avg_accuracy

# Print the average accuracies for each subject
for subject, avg_accuracy in average_accuracies.items():
    print(f"Subject {subject}: Average Accuracy = {avg_accuracy:.4f}")

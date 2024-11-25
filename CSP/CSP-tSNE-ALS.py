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
from sklearn.manifold import TSNE
import seaborn as sns

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


del data, fold, fold_data, fold_idx, fold_info, i, L_data, length, max_length, maxlen, merged_folds, num_channels, num_samples, num_trials, padded_fold_data, R_data, subj_idx, subject
#%%

# Create a new variable to store the transformed data
transformed_data = {}

# Iterate over each subject in padded_data
for subject in padded_data.keys():
    # Initialize a new dictionary for each subject to store transformed sessions
    transformed_data[subject] = {
        'transformed_sessions': [],
        'labels': []
    }
    
    # Extract the sessions data and labels
    sessions = padded_data[subject]
    
    # Extract the data and labels for the first session
    first_session_data = sessions[0]['data'].astype(np.float64)
    first_session_labels = sessions[0]['labels']
    
    # Initialize the CSP model
    csp = CSP(n_components=4)  # Adjust n_components as needed
    
    # Fit the CSP model on the first session's data
    csp.fit(first_session_data, first_session_labels)
    
    # Apply the CSP model to transform the first session's data
    transformed_first_session = csp.transform(first_session_data)
    
    # Store the transformed first session data and labels
    transformed_data[subject]['transformed_sessions'].append(transformed_first_session)
    transformed_data[subject]['labels'].append(first_session_labels)
    
    # Iterate over the remaining 4 sessions
    for i in range(1, 4):
        session_data = sessions[i]['data'].astype(np.float64)
        session_labels = sessions[i]['labels']
        
        # Apply the fitted CSP model to transform the session's data
        transformed_session = csp.transform(session_data)
        
        # Store the transformed session data and labels
        transformed_data[subject]['transformed_sessions'].append(transformed_session)
        transformed_data[subject]['labels'].append(session_labels)

#%%


# Number of folds (sessions)
num_folds = 4

# Iterate over each subject in transformed_data
for subject in transformed_data.keys():
    # Create a new figure for each subject
    fig, axes = plt.subplots(1, num_folds, figsize=(24, 8), sharex=True, sharey=True)  # Increase figsize for larger plots
    
    # Iterate over each fold (session) for the current subject
    for fold_idx in range(num_folds):
        # Access the 'transformed_sessions' and 'labels' data for the current fold
        x_data = transformed_data[subject]['transformed_sessions'][fold_idx]  # Shape should be (num_samples, num_features)
        y_labels = transformed_data[subject]['labels'][fold_idx]
        
        # Apply t-SNE to the data
        tsne = TSNE(n_components=2, random_state=0, perplexity=30.0)
        tsne_results = tsne.fit_transform(x_data)  # No reshaping needed since x_data is already (num_samples, num_features)
        
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

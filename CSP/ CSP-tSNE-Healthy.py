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

#%%%%%%%%%%%% SHU Dataset - CSP %%%%%%%%%%%%%%%%

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

#% Split Left and Rights

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

#%
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

#%
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

del filtered_data_split
#%%
data = {}

# Iterate over each subject
for subject in fold_data.keys():
    # Initialize lists to store the merged sessions and labels
    merged_sessions = []
    session_labels = []
    
    # Get the number of sessions (assuming L and R have the same number of sessions)
    num_sessions = len(fold_data[subject]['L'])
    
    # Iterate over each session index
    for i in range(num_sessions):
        # Retrieve the L and R session arrays
        L_session = fold_data[subject]['L'][i]
        R_session = fold_data[subject]['R'][i]
        
        # Merge the L and R session arrays for the current session index
        merged_session = np.concatenate((L_session, R_session), axis=0)
        
        # Generate labels for the current session
        # Assuming 0 for L data and 1 for R data
        labels = np.concatenate((np.zeros(L_session.shape[0]), np.ones(R_session.shape[0])))
        
        # Append the merged session and labels to their respective lists
        merged_sessions.append(merged_session)
        session_labels.append(labels)
    
    # Store the merged sessions and labels in the `data` dictionary
    data[subject] = {
        'data': merged_sessions,
        'labels': session_labels
    }
    
del labels,L_session,R_session,session_labels,subject,num_sessions,merged_sessions,merged_session, i, fold_data

# Create a new variable to store the transformed data
transformed_data = {}

# Iterate over each subject
for subject in data.keys():
    # Initialize a new dictionary for each subject to store transformed sessions
    transformed_data[subject] = {
        'transformed_sessions': [],
        'labels': []
    }
    
    # Extract the data and labels for the first session
    first_session_data = data[subject]['data'][0].astype(np.float64)
    first_session_labels = data[subject]['labels'][0].astype(np.float64)
    
    # Initialize the CSP model
    csp = CSP(n_components=4)  # Adjust n_components as needed
    
    # Fit the CSP model on the first session's data
    csp.fit(first_session_data, first_session_labels)
    
    # Apply the CSP model to transform the first session's data (for consistency)
    transformed_first_session = csp.transform(first_session_data)
    
    # Store the transformed first session data and labels
    transformed_data[subject]['transformed_sessions'].append(transformed_first_session)
    transformed_data[subject]['labels'].append(first_session_labels)
    
    # Now, iterate over the next 4 sessions (i.e., session indices 1 to 4)
    for i in range(1, 5):
        session_data = data[subject]['data'][i].astype(np.float64)
        session_labels = data[subject]['labels'][i].astype(np.float64)
        
        # Apply the fitted CSP model to transform the session's data
        transformed_session = csp.transform(session_data)
        
        # Store the transformed session data and labels
        transformed_data[subject]['transformed_sessions'].append(transformed_session)
        transformed_data[subject]['labels'].append(session_labels)

del csp, first_session_data, first_session_labels, session_data, session_labels, subject, transformed_first_session, transformed_session

#%% t-SNE


# Define the number of sessions
num_sessions = 5

# Iterate over each subject
for subject in data.keys():
    # Create a new figure for each subject
    fig, axes = plt.subplots(1, num_sessions, figsize=(20, 4), sharex=True, sharey=True)
    
    # Extract transformed sessions data and labels
    sessions_data = transformed_data[subject]['transformed_sessions']
    sessions_labels = transformed_data[subject]['labels']
    
    # Iterate over each session
    for session_idx in range(num_sessions):
        # Get the transformed data and labels for the current session
        session_data = sessions_data[session_idx]
        session_labels = sessions_labels[session_idx]
        
        # Apply t-SNE to the session data
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(session_data)
        
        # Plot the t-SNE results
        ax = axes[session_idx]
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=session_labels, cmap='viridis', s=10)
        ax.set_title(f'Session {session_idx+1}')
        ax.axis('off')  # Hide axis
    
    # Optionally, add a colorbar to the last subplot
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Labels')
    
    # Set a main title for the figure
    fig.suptitle(f'Subject {subject}')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the colorbar
    plt.show()
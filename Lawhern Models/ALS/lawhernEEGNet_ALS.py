import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from os.path import join as pjoin
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import mne
from mne import io
import os
import scipy.signal as sig

# EEGNet-specific imports
from EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

#%%

def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = sig.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sig.sosfiltfilt(sos, data,axis=0)
    return filtered_data

def aggregate_eeg_data(S1, chunk_size=1024):
    """
    Aggregate EEG data by selecting the middle chunk of fixed size from each trial and 
    merging 'L' and 'R' trials sequentially.

    Parameters:
        S1 (dict): Dictionary containing EEG data for each class. Keys are class labels,
                   values are arrays of shape (2, num_samples, num_channels), where the first dimension
                   corresponds to EEG data (index 0) and frequency data (index 1).
        chunk_size (int): The size of each chunk to be extracted from the middle of the trial.

    Returns:
        data (ndarray): Aggregated EEG data with shape (num_trials * 2, chunk_size, numElectrodes),
                        where 2 represents the 'L' and 'R' classes.
        labels (ndarray): Labels for each chunk with shape (num_trials * 2,)
                          where 0 represents 'L' and 1 represents 'R'.
    """
    numElectrodes = S1['L'][0, 1].shape[1]

    # Initialize lists to store aggregated EEG data and labels
    data_list = []
    labels_list = []

    # Loop through each trial and select the middle chunk of 'L' and 'R' trials
    for i in range(S1['L'].shape[1]):
        # Process 'L' trial
        l_trial = S1['L'][0, i]
        l_num_samples = l_trial.shape[0]

        if l_num_samples >= chunk_size:
            # Calculate the start and end indices for the middle chunk
            l_start = (l_num_samples - chunk_size) // 2
            l_end = l_start + chunk_size
            l_middle_chunk = l_trial[l_start:l_end, :]  # Select the middle chunk
            data_list.append(l_middle_chunk)
            labels_list.append(0)  # Label for 'L'

        # Process 'R' trial
        r_trial = S1['R'][0, i]
        r_num_samples = r_trial.shape[0]

        if r_num_samples >= chunk_size:
            # Calculate the start and end indices for the middle chunk
            r_start = (r_num_samples - chunk_size) // 2
            r_end = r_start + chunk_size
            r_middle_chunk = r_trial[r_start:r_end, :]  # Select the middle chunk
            data_list.append(r_middle_chunk)
            labels_list.append(1)  # Label for 'R'

    # Convert lists to numpy arrays
    data = np.stack(data_list, axis=0)  # Shape: (num_trials * 2, chunk_size, numElectrodes)
    labels = np.array(labels_list)      # Shape: (num_trials * 2,)

    return data, labels


# Function to save the results
def save_best_accuracies(best_accuracies, save_path='lawhernEEGNetHealthy.npy'):
    np.save(save_path, best_accuracies)

#%%

import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from time import time

# Data directory
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Year 2\DeepLearning-on-ALS-MI-Data\Graphs\Data'

# Define the subject numbers
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
fs = 256
idx = ['L', 'R']

# Initialize dictionary to store results
results = {}

for subject_number in subject_numbers:
    # Load subject data
    mat_fname = os.path.join(data_dir, f'OGFS{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    subject_data = mat_contents[f'Subject{subject_number}'][:, :-1]

    # Preprocess data (bandpass filtering)
    for key in idx:
        for i in range(subject_data.shape[1]):
            subject_data[key][0, i] = bandpass(subject_data[key][0, i], [8, 30], fs)

    # Aggregate and format EEG data
    data, labels = aggregate_eeg_data(subject_data, chunk_size=256 * 3)

    trials = data.shape[0]
    chans = data.shape[2]
    samples = data.shape[1]
    kernels = 1

    data = data.reshape(trials, chans, samples, kernels)  # N x C x T x K
    labels = labels.reshape(-1, 1)
    labels = OneHotEncoder(sparse_output=False).fit_transform(labels)

    # Split the data into four equal sessions (no shuffling)
    split_size = trials // 4
    session_data = [data[i * split_size:(i + 1) * split_size] for i in range(4)]
    session_labels = [labels[i * split_size:(i + 1) * split_size] for i in range(4)]

    # Train on Session 1
    train_data = session_data[0]
    train_labels = session_labels[0]

    # Initialize EEGNet model
    model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
                   dropoutRate=0.5, kernLength=128, F1=8, D=2, F2=16,
                   dropoutType='Dropout')
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # Initialize variables to track the best accuracies for each session
    best_accuracies = {f'1v{session_idx + 1}': 0 for session_idx in range(1, 4)}
    subject_results = {'train_time': 0, 'best_session_accuracies': {}, 'inference_times': {}}
    
    print(f"Training model for Subject {subject_number}...")
    
    start_train_time = time()
    
    # Train the model for a specific number of epochs
    for epoch in range(500):
        print(f"Epoch {epoch + 1}/500")
    
        # Train for one epoch
        model.fit(train_data, train_labels, batch_size=32, epochs=1, verbose=2)
    
        # Test on Sessions 2, 3, 4 after each epoch
        for session_idx in range(1, 4):  # Test on Sessions 2, 3, 4
            test_data = session_data[session_idx]
            test_labels = session_labels[session_idx]
    
            # Measure inference time
            start_inference_time = time()
            probs = model.predict(test_data)
            inference_time = time() - start_inference_time
    
            preds = probs.argmax(axis=-1)
            acc = np.mean(preds == test_labels.argmax(axis=-1))
    
            # Update best accuracy for the session if the current accuracy is higher
            session_key = f'1v{session_idx + 1}'
            if acc > best_accuracies[session_key]:
                best_accuracies[session_key] = acc
    
            # Store the latest inference time (could also store epoch-specific times if needed)
            subject_results['inference_times'][session_key] = inference_time / test_data.shape[0]
    
            print(f"Subject {subject_number}, Epoch {epoch + 1}, Test on Session {session_key}: "
                  f"Accuracy = {acc:.4f}, Best Accuracy = {best_accuracies[session_key]:.4f}, "
                  f"Avg Inference Time = {inference_time / test_data.shape[0]:.4f} seconds")
    
    # Record total training time
    total_train_time = time() - start_train_time
    subject_results['train_time'] = total_train_time
    
    print(f"Training complete. Total training time: {total_train_time:.4f} seconds.")
    
    # Store the best accuracies for all sessions
    subject_results['best_session_accuracies'] = best_accuracies
    results[f'Subject_{subject_number}'] = subject_results

# Print final results for the subject
print(f"Subject {subject_number} Final Results:")
for session_key, acc in best_accuracies.items():
    print(f"  Best Accuracy for {session_key}: {acc:.4f}")


# Save results to a file
save_path = os.path.join(os.getcwd(), 'session_accuracies_and_timings.npy')
np.save(save_path, results)
print(f"Results saved to {save_path}")

# Print summary of results
for subject, res in results.items():
    print(f"Results for {subject}:")
    print(f"  Total Training Time: {res['train_time']:.4f} seconds")
    for session, acc in res['best_session_accuracies'].items():
        print(f"  {session}: Accuracy = {acc:.4f}, Avg Inference Time = {res['inference_times'][session]:.4f} seconds")

# Prepare results for visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_for_viz = []

for subject, res in results.items():
    for session, acc in res['best_session_accuracies'].items():
        data_for_viz.append({'Subject': subject, 'Train-Test Pair': session, 'Accuracy': acc})

# Convert to DataFrame
df = pd.DataFrame(data_for_viz)

# Generate the Boxplot
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
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Calculate and print average training and inference times across all subjects
all_training_times = [res['train_time'] for res in results.values()]
all_inference_times = [
    np.mean(list(res['inference_times'].values())) for res in results.values()
]

avg_training_time = np.mean(all_training_times)
avg_inference_time = np.mean(all_inference_times)

print(f"Average Training Time (500 epochs): {avg_training_time:.2f} seconds")
print(f"Average Inference Time per trial: {avg_inference_time:.6f} seconds\n")

# Print detailed results per subject
for subject, res in results.items():
    print(f"Subject {subject}:")
    print(f"  Training Time (500 epochs): {res['train_time']:.2f} seconds")
    print("  Inference Times (seconds/trial):")
    for pair, inf_time in res['inference_times'].items():
        print(f"    {pair}: {inf_time:.6f} seconds/trial")
    print("  Pairwise Accuracies:")
    for pair, accuracy in res['best_session_accuracies'].items():
        print(f"    {pair}: {accuracy:.4f}")
    print("\n")

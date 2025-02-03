import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, hilbert, coherence
from scipy.linalg import norm
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm

# ---------------------------
# Signal Processing Functions
# ---------------------------
def bandpasscfc(data, freq_range, sample_rate=256, order=5):
    """
    Bandpass filter the data within a given frequency range.
    
    Parameters:
        data (ndarray): Input time-series data.
        freq_range (list or tuple): [low_freq, high_freq].
        sample_rate (int): Sampling frequency.
        order (int): Order of the Butterworth filter.
        
    Returns:
        ndarray: Bandpass-filtered data.
    """
    nyquist = 0.5 * sample_rate
    low = freq_range[0] / nyquist
    high = freq_range[1] / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def cfcfcn(eegData, fs=256):
    """
    Compute cross-frequency coupling (CFC) using the phase of the Mu band and 
    the amplitude of the Beta band, then normalize the resulting matrix to [0,1].
    
    Parameters:
        eegData (ndarray): EEG data of shape (time, electrodes).
        fs (int): Sampling frequency.
        
    Returns:
        ndarray: Normalized symmetric CFC matrix.
    """
    numElectrodes = eegData.shape[1]
    cfcMatrix = np.zeros((numElectrodes, numElectrodes))
    
    # Filter for Mu (8-13 Hz) and Beta (13-30 Hz) bands
    mu_data = bandpasscfc(eegData, [8, 13], sample_rate=fs)
    beta_data = bandpasscfc(eegData, [13, 30], sample_rate=fs)
    
    # Extract phase from Mu band and amplitude from Beta band
    mu_phase = np.angle(hilbert(mu_data, axis=0))
    beta_amplitude = np.abs(hilbert(beta_data, axis=0))
    
    # Compute PAC-based CFC values
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            pac_value = np.abs(np.mean(beta_amplitude[:, electrode2] * 
                                       np.exp(1j * mu_phase[:, electrode1])))
            cfcMatrix[electrode1, electrode2] = pac_value
            cfcMatrix[electrode2, electrode1] = pac_value  # Ensure symmetry

    # Normalize CFC matrix to 0-1 range
    cfc_min = np.min(cfcMatrix)
    cfc_max = np.max(cfcMatrix)
    
    if cfc_max != cfc_min:  # Avoid division by zero
        cfcMatrix = (cfcMatrix - cfc_min) / (cfc_max - cfc_min)

    return cfcMatrix


def mscfcn(eegData, fs=256, nperseg=256):
    """
    Compute mean squared coherence (MSC) between all pairs of electrodes.
    
    Parameters:
        eegData (ndarray): EEG data of shape (time, electrodes).
        fs (int): Sampling frequency.
        nperseg (int): Length of each segment for coherence calculation.
        
    Returns:
        ndarray: Symmetric MSC matrix.
    """
    numElectrodes = eegData.shape[1]
    mscMatrix = np.zeros((numElectrodes, numElectrodes))
    
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            f, Cxy = coherence(eegData[:, electrode1], eegData[:, electrode2], 
                               fs=fs, nperseg=nperseg)
            msc_val = np.mean(Cxy)
            mscMatrix[electrode1, electrode2] = msc_val
            mscMatrix[electrode2, electrode1] = msc_val
    return mscMatrix

def cmifcn(eegData):
    """
    Compute conditional mutual information (CMI) between pairs of electrodes.
    
    Parameters:
        eegData (ndarray): EEG data of shape (time, electrodes).
        
    Returns:
        ndarray: Symmetric CMI matrix.
        
    Note: mutual_info_regression returns an array; we extract the scalar value.
    """
    numElectrodes = eegData.shape[1]
    cmiMatrix = np.zeros((numElectrodes, numElectrodes))
    
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            signal1 = eegData[:, electrode1]
            signal2 = eegData[:, electrode2]
            # mutual_info_regression returns an array; take the first (and only) element
            cmi_value = mutual_info_regression(signal1.reshape(-1, 1), signal2)[0]
            cmiMatrix[electrode1, electrode2] = cmi_value
            cmiMatrix[electrode2, electrode1] = cmi_value
    return cmiMatrix

def plvfcn(eegData):
    """
    Compute Phase Locking Value (PLV) between all pairs of electrodes.
    
    Parameters:
        eegData (ndarray): EEG data of shape (time, electrodes).
        
    Returns:
        ndarray: Symmetric PLV matrix.
    """
    numElectrodes = eegData.shape[1]
    numTimeSteps = eegData.shape[0]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(hilbert(eegData[:, electrode1]))
            phase2 = np.angle(hilbert(eegData[:, electrode2]))
            phase_difference = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_difference)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    return plvMatrix

# ---------------------------
# Utility Functions
# ---------------------------
def frobenius_norm(matrix1, matrix2):
    """
    Compute the Frobenius norm of the difference between two matrices.
    
    Parameters:
        matrix1, matrix2 (ndarray): Input matrices.
        
    Returns:
        float: Frobenius norm.
    """
    return norm(matrix1 - matrix2, 'fro')

def compute_all_measures(subject_data, fs=256):
    """
    For a given subject, compute the connectivity matrices (PLV, MSC, CFC, CMI)
    for both Left and Right classes over all trials.
    
    Parameters:
        subject_data (dict): Dictionary with keys 'L' and 'R'. Each contains
                             an array of trials where a trial is accessed as 
                             subject_data[side][0, trial].
        fs (int): Sampling frequency.
        
    Returns:
        dict: Nested dictionary with measures as keys and each containing a dict
              with keys 'L' and 'R'. Each entry is an array of shape 
              (nElectrodes, nElectrodes, nTrials).
    """
    measures = {'PLV': {}, 'MSC': {}, 'CFC': {}, 'CMI': {}}
    for side in ['L', 'R']:
        nTrials = subject_data[side].shape[1]
        nElectrodes = subject_data[side][0, 0].shape[1]
        measures['PLV'][side] = np.zeros((nElectrodes, nElectrodes, nTrials))
        measures['MSC'][side] = np.zeros((nElectrodes, nElectrodes, nTrials))
        measures['CFC'][side] = np.zeros((nElectrodes, nElectrodes, nTrials))
        measures['CMI'][side] = np.zeros((nElectrodes, nElectrodes, nTrials))
        for t in range(nTrials):
            eeg_data = subject_data[side][0, t]
            measures['PLV'][side][:, :, t] = plvfcn(eeg_data)
            measures['MSC'][side][:, :, t] = mscfcn(eeg_data, fs=fs)
            measures['CFC'][side][:, :, t] = cfcfcn(eeg_data, fs=fs)
            measures['CMI'][side][:, :, t] = cmifcn(eeg_data)
    return measures

def compute_successive_diff(matrix_stack):
    """
    Compute the average Frobenius norm between successive trial connectivity 
    matrices.
    
    Parameters:
        matrix_stack (ndarray): 3D array of shape (nElectrodes, nElectrodes, nTrials).
        
    Returns:
        float: Average Frobenius norm difference between successive trials.
    """
    diffs = []
    nTrials = matrix_stack.shape[2]
    for i in range(nTrials - 1):
        diff = frobenius_norm(matrix_stack[:, :, i], matrix_stack[:, :, i + 1])
        diffs.append(diff)
    return np.mean(diffs) if diffs else np.nan

def compute_inter_measure_diff(plv, msc, cfc, cmi):
    """
    For each trial, compute the Frobenius norm differences between each pair of 
    measures and return the average over trials.
    
    Parameters:
        plv, msc, cfc, cmi (ndarray): 3D arrays of connectivity matrices of shape 
                                     (nElectrodes, nElectrodes, nTrials).
                                     
    Returns:
        dict: A dictionary with keys corresponding to measure pairs and values 
              being the average Frobenius norm difference over trials.
    """
    nTrials = plv.shape[2]
    diffs = {"PLV_MSC": [], "PLV_CFC": [], "PLV_CMI": [],
             "MSC_CFC": [], "MSC_CMI": [], "CFC_CMI": []}
    for t in range(nTrials):
        diffs["PLV_MSC"].append(frobenius_norm(plv[:, :, t], msc[:, :, t]))
        diffs["PLV_CFC"].append(frobenius_norm(plv[:, :, t], cfc[:, :, t]))
        diffs["PLV_CMI"].append(frobenius_norm(plv[:, :, t], cmi[:, :, t]))
        diffs["MSC_CFC"].append(frobenius_norm(msc[:, :, t], cfc[:, :, t]))
        diffs["MSC_CMI"].append(frobenius_norm(msc[:, :, t], cmi[:, :, t]))
        diffs["CFC_CMI"].append(frobenius_norm(cfc[:, :, t], cmi[:, :, t]))
    # Average differences over trials for each measure pair
    avg_diffs = {key: np.mean(val) for key, val in diffs.items()}
    return avg_diffs

# ---------------------------
# Main Data Processing Loop
# ---------------------------
data_dir = r'C:\Users\uceerjp\Desktop\PhD\Multi-session Data\OG_Full_Data'
subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]

# Storage for metrics:
# 1. Successive trial differences per measure (PLV, MSC, CFC, CMI)
successive_metrics = {measure: {'L': [], 'R': []} for measure in ['PLV', 'MSC', 'CFC', 'CMI']}
# 2. Inter-measure differences per measure pair
inter_measure_metrics = {pair: {'L': [], 'R': []} for pair in 
                         ["PLV_MSC", "PLV_CFC", "PLV_CMI", "MSC_CFC", "MSC_CMI", "CFC_CMI"]}

for subject_number in tqdm(subject_numbers, desc="Processing Subjects"):
    mat_fname = os.path.join(data_dir, f'S{subject_number}.mat')
    mat_contents = sio.loadmat(mat_fname)
    # Assume the subject data is stored under the key f'Subject{subject_number}' 
    # and that the last entry is to be excluded.
    subject_data = mat_contents[f'Subject{subject_number}'][:, :-1]
    
    # Compute connectivity matrices for all measures
    measures = compute_all_measures(subject_data, fs=256)
    
    for side in ['L', 'R']:
        # Compute successive trial differences for each measure
        for measure in ['PLV', 'MSC', 'CFC', 'CMI']:
            diff_val = compute_successive_diff(measures[measure][side])
            successive_metrics[measure][side].append(diff_val)
        
        # Compute inter-measure differences (averaged over trials) for this class
        inter_diffs = compute_inter_measure_diff(measures['PLV'][side],
                                                 measures['MSC'][side],
                                                 measures['CFC'][side],
                                                 measures['CMI'][side])
        for pair, value in inter_diffs.items():
            inter_measure_metrics[pair][side].append(value)

# ---------------------------
# Reporting Results
# ---------------------------
print("\nSuccessive Trial Differences (Frobenius Norm) per measure:")
for measure in ['PLV', 'MSC', 'CFC', 'CMI']:
    for side in ['L', 'R']:
        values = np.array(successive_metrics[measure][side])
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{measure} {side}: {mean_val:.4f} ± {std_val:.4f}")

print("\nInter-Measure Differences (Frobenius Norm) per pair:")
for pair in ["PLV_MSC", "PLV_CFC", "PLV_CMI", "MSC_CFC", "MSC_CMI", "CFC_CMI"]:
    for side in ['L', 'R']:
        values = np.array(inter_measure_metrics[pair][side])
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{pair} {side}: {mean_val:.4f} ± {std_val:.4f}")

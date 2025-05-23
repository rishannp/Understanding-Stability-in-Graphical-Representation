# Enhancing Cross-Session Motor Imagery Classification Using Graphical Models in ALS Patients

This repository contains the official implementation for our study:

> **"Enhancing Cross-Session Motor Imagery Classification Using Graphical Models in ALS Patients"**  
> Rishan Patel, Barney Bryson, Tom Carlson, Dai Jiang, and Andreas Demosthenous  
> *Under Review in tNSRE*  

---

## 🧠 Overview

This work addresses the challenge of non-stationarity in EEG-based Brain-Computer Interfaces (BCIs), particularly in longitudinal use cases for ALS patients. We explore the effectiveness of various graph-theoretic EEG representations; Phase Locking Value (PLV), Magnitude Squared Coherence (MSC), Cross Frequency Coupling (CFC), and Conditional Mutual Information (CMI) within a Graph Attention Network (GAT) framework to improve cross-session motor imagery classification.

---

## 📊 Key Results

- **PLV-GAT** outperforms conventional methods (CSP, EEGNet, DeepConvNet) in both ALS and healthy cohorts.
- Graph-based models exhibit enhanced robustness to session-to-session signal variability.
- Ablation studies identify subject-specific electrode configurations that further boost performance.
- PLV emerges as a computationally efficient, stable, and explainable candidate for real-time BCI.
---

## 🧬 Dataset

We evaluate our models on two datasets:

1. **SHU Dataset** — 25 healthy subjects, 5 sessions each, 32-channel EEG.
2. **ALS Dataset** — 8 ALS patients, 4 sessions each, 19-channel EEG, recorded using g.USBamp and BCI2000.

Details and access:  
- ALS Dataset DOI: [https://doi.org/10.5522/04/28156016.v1](https://doi.org/10.5522/04/28156016.v1)  
- SHU Dataset: Referenced from [36] in the paper.

---

## 🧰 Requirements

- Python 3.8+
- PyTorch 2.x
- PyTorch Geometric 2.5.0
- NumPy, SciPy, Scikit-learn
- Matplotlib, Seaborn (for plots)


## Deadline for implementation: December 31st 2024

Model our paper after: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8961150
## Aim: 
Investigate what processing metrics are good, what is not good and why. Why do graph features work favorably in patient/lower performing subject data? 
1. Graph features work favorably at intersession classification
2. Why is this the case? How can these be implemented in real-time BCIs

#followup could check the evolution of graphs over 1fs per segment MI.

#followup check the t-SNE plots for better subject specific perplexities.

## Hypothesis: 
*Graphical representations of EEG are more stable a feature inter-session than alternative methods like CSP, we believe it is because Graph representations are more robust to signal changes because they do not rely on time domain features but also exploit relationships to neighboring channels*  

## Experiment: 
Using Different Connectivity matrices, build graphs per trial, take the average graph per trial and check the inter-session changes over time. 

Use the ALS and Healthy dataset to understand whether graphical representations benefit from greater stability. We can prove non-stationarity with mean and variance changes per trial, and then compare all of the models to CSP/some of the deep learning models in the nature data descriptor for SHU. The SHU Dataset already has work done on it, and it is easy to draw comparison as their code is accessible. 

From this, we will see whether Graph is beneficial, and can improve stability, and give insights into different representations which offer the highest levels of stability.

The Evidence for this is Accuracies and t-SNE plots. It is clear that intersession representations are significantly enhanced.

Ziyue Idea: Plot t-SNE per trial, so its like a plot time series. Then find the Euclidian distance between each successive sessions graphs to quantify stability. this is better as i can do it for both CSP and Graphs:

- **Cluster Stability**: If the t-SNE plots for the subsequent sessions show stable clusters (similar to the first session), it suggests that the EEG data is relatively stationary across sessions.
- **Cluster Drift**: If I observe that the clusters drift, deform, or merge across sessions, this indicates non-stationarity. Such changes may arise from factors like electrode shift, changes in the subject’s cognitive state, or other physiological variations.
- **Overlap Increase**: An increase in the overlap between clusters corresponding to different classes across sessions could suggest a degradation in the discriminability of the features, potentially impacting classification performance.

Attention Weights Heatmaps. 

The question of: Why is this better? Is yet to be answered. 
There is perhaps a more nuanced answer, and requires significant thought. Once everything is finished, I should #followup on this.

## Methods: 

Use SHU Multi-session dataset and Penn State Dataset

- Show the Mean and Variance change over time in all the data. This will show the data is non-stationary as a base evidence. 
- Comput tSNE for each session. 

Training Testing Protocol:
Take 1/5 Sessions for SHU and test against other 4 one by one
Take 1/4 sessions for ALS and test against other 3 one by one

1. A standard control method would be using CSP to classify
2. Then compare multiple different connectivity matrices on GAT classification
3. Test a fully connected graph with Band Power
4. Test a fully connected graph with Connectivity Matrices

Connectivity Matrices: 10.1016/j.compbiomed.2011.06.020
- [ ] Cross Frequency Coupling - 
- [ ] Magnitude Squared Coherence - Linear
- [ ] Phase Locking Value - a stationarity independent measure such as PLV can be used  - Referecnce above - nonlinear
- [ ] Cross Mutual Information - information based
- [x] Wavelet Coherence - nonlinear
- [x] Generalized Synchronization - nonlinar

Considerations: 

Model based approaches are based on well-defined biophysical models of neuronal dynamics. Choosing a best model, and define experiment with different parameters to test a hypothesis. 
The number of parameters is the drawback of this methodology. 
Data driven methods do not assume any underlying relationship but assess connections when no a-priori knowlegde is available. 

Stationarity: Most methods assume stationarity. For a process to be stationary, the mean, variance and auto-correlation structure cannot change over time. Generally, EEG is considered multi-variate gaussian, even if mean and covariance change. EEG is quasi-stationary.  Stationary Subspace Analysis could help decomopose. 

Chaotic systems appeaer to be noisy, which is deterministic. Although non-linear measures can identify the interdependencies, they are susceptible to noise. The requirement of long stationary epochs, even though neurons are highly non-linear (in theory), strong evidence of chaos has not been found in EEG. Wide consensus that EEG is not chaotic. Linear measures are more robust and perform well even in non-linear cases. **Non linear and linear should be used in conjunction.** 

![[Pasted image 20240812155332.png]]
Sakkakis 2011

- #futurework Explore how Cross Frequency Coupling can impact accuracies on different frequencies
## To do:
- [ ] Work on introduction. 
- Redo all of the models, but for four items:
- [ ] **Step 1: Add ICC recording**
- [ ] **Step 5: Add t-SNE of all final layers?** 
- [ ] **Step 6: Attention Scores**
- [ ] Explainability for AI

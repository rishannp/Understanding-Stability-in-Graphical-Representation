import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style for plots
sns.set(style="whitegrid", context="talk")

# ALS Channel Locations
ALS_ch_names = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']
ALS_coords = [
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

# SHU Channel Locations
SHU_ch_names = ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 
                'CZ', 'C3', 'C4', 'T3', 'T4', 'A1', 'A2', 'CP1', 'CP2', 'CP5', 'CP6', 
                'PZ', 'P3', 'P4', 'T5', 'T6', 'PO3', 'PO4', 'OZ', 'O1', 'O2']
SHU_coords = [
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
    [3.68e-15, 90.1, -60.1],                  # A1
    [3.68e-15, -90.1, -60.1],                 # A2
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


# Set Seaborn style for plots
sns.set(style="whitegrid", context="talk")

# Plotting ALS Channels (Rotated 90 degrees anticlockwise)
fig1, ax1 = plt.subplots(figsize=(10, 8))
sns.scatterplot(x=[coord[1] for coord in ALS_coords], y=[-coord[0] for coord in ALS_coords],  # Swap x and y, invert new y
                hue=ALS_ch_names, palette="coolwarm", s=100, edgecolor='k', ax=ax1, legend=False)

# Adjust label positions to avoid overlap (rotated positions)
for i, txt in enumerate(ALS_ch_names):
    ax1.text(ALS_coords[i][1], -ALS_coords[i][0] - 0.05, txt, fontsize=12, ha='center', color='black')

# Flip x-axis and y-axis
ax1.invert_xaxis()
ax1.invert_yaxis()

ax1.set_title('ALS Electrode Array', fontsize=16)
ax1.set_aspect('equal', 'box')

# Plotting SHU Channels (Rotated 90 degrees anticlockwise)
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.scatterplot(x=[coord[1] for coord in SHU_coords], y=[-coord[0] for coord in SHU_coords],  # Swap x and y, invert new y
                hue=SHU_ch_names, palette="viridis", s=100, edgecolor='k', ax=ax2, legend=False)

# Adjust label positions to avoid overlap (rotated positions)
for i, txt in enumerate(SHU_ch_names):
    ax2.text(SHU_coords[i][1], -SHU_coords[i][0] - 4, txt, fontsize=12, ha='center', color='black')

# Flip x-axis and y-axis
ax2.invert_xaxis()
ax2.invert_yaxis()

ax2.set_title('SHU Electrode Array', fontsize=16)
ax2.set_aspect('equal', 'box')

# Show the plots
plt.tight_layout()
plt.show()

#%%

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Set Seaborn style for plots
sns.set(style="whitegrid", context="talk")

# 3D Plot for ALS Channels (Rotated 90 degrees anticlockwise)
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')

# 3D scatter plot for ALS channels
scatter1 = ax1.scatter(
    [coord[1] for coord in ALS_coords], 
    [-coord[0] for coord in ALS_coords], 
    [coord[2] for coord in ALS_coords],  # Use the z-coordinate
    c=range(len(ALS_ch_names)), cmap="coolwarm", s=100, edgecolors='k'
)

# Annotate points with electrode names
for i, txt in enumerate(ALS_ch_names):
    ax1.text(ALS_coords[i][1], -ALS_coords[i][0], ALS_coords[i][2] - 0.05, txt, fontsize=12, ha='center', color='black')

# Invert axes to match the previous 2D orientation
ax1.invert_xaxis()
ax1.invert_yaxis()

ax1.set_title('ALS Electrode Array (3D Plot)', fontsize=16)
ax1.set_xlabel('Y Coordinate')  # Y is the swapped coordinate
ax1.set_ylabel('-X Coordinate')
ax1.set_zlabel('Z Coordinate')
ax1.set_aspect('auto')

# 3D Plot for SHU Channels (Rotated 90 degrees anticlockwise)
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')

# 3D scatter plot for SHU channels
scatter2 = ax2.scatter(
    [coord[1] for coord in SHU_coords], 
    [-coord[0] for coord in SHU_coords], 
    [coord[2] for coord in SHU_coords],  # Use the z-coordinate
    c=range(len(SHU_ch_names)), cmap="viridis", s=100, edgecolors='k'
)

# Annotate points with electrode names
for i, txt in enumerate(SHU_ch_names):
    ax2.text(SHU_coords[i][1], -SHU_coords[i][0], SHU_coords[i][2] - 4, txt, fontsize=12, ha='center', color='black')

# Invert axes to match the previous 2D orientation
ax2.invert_xaxis()
ax2.invert_yaxis()

ax2.set_title('SHU Electrode Array (3D Plot)', fontsize=16)
ax2.set_xlabel('Y Coordinate')  # Y is the swapped coordinate
ax2.set_ylabel('-X Coordinate')
ax2.set_zlabel('Z Coordinate')
ax2.set_aspect('auto')

# Show the plots
plt.tight_layout()
plt.show()

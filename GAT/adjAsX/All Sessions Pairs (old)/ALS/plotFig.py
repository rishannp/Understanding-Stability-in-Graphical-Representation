import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Path to your .pkl file
file_path = 'CMIasX.pkl'

# Open the .pkl file and load the content
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Now, 'data' contains the deserialized object
print(data)

# Assume 'data' is the same as 'all_subjects_accuracies'
all_subjects_accuracies = data

# Prepare data
plot_data = []
for subject, folds in all_subjects_accuracies.items():
    for fold_data in folds:
        train_test_pair = fold_data['fold_name']
        accuracy = fold_data['optimal']  # Using 'optimal' accuracy here, you can change it to 'mean', 'high', or 'low'
        plot_data.append({
            'Subject': subject,  # E.g., 'S1', 'S2', etc.
            'Train-Test Pair': train_test_pair,
            'Accuracy': accuracy
        })

df = pd.DataFrame(plot_data)

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


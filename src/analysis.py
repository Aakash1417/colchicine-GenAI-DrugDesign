import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("experiment/staged_learning_1.csv")

# Check if 'step' and 'Score' columns are present
if 'step' not in df.columns or 'Score' not in df.columns:
    raise ValueError("Make sure the file contains 'step' and 'Score' columns")

# Group by 'step' and extract the scores
grouped_scores = df.groupby('step')['Score'].apply(list)

# Prepare data for boxplot
data = [grouped_scores[step] for step in sorted(grouped_scores.index)]

# Plotting
plt.figure(figsize=(12, 6))
plt.boxplot(data, labels=sorted(grouped_scores.index))
plt.xlabel('Step')
plt.ylabel('Score')
plt.title('Distribution of Scores by Step')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/boxplot_scores_by_step.png', dpi=300)

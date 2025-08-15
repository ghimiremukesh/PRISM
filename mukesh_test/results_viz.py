#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns
import ipdb 

# In[6]:


# Update this path to your results file from the logprobs script
results_file = 'self_certainty_sc2.json'


# In[8]:


# Load the results
with open(results_file, 'r') as f:
    results = json.load(f)

# Convert to DataFrame for easier analysis
df = pd.DataFrame(results)




# In[9]:

# ipdb.set_trace()
# Add mean self-certainty for each example
df['self_certainty'] = df['confidence_list'].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)


correct_mask = np.array([df['specifics'][i]['extracted_golds'] == df['specifics'][i]['extracted_predictions'] for i in range(len(df))])
correct_df = df[correct_mask]
incorrect_df = df[~correct_mask]

# ## Basic Statistics

# In[ ]:


# Count of correct and incorrect answers
correct_count = len(correct_df)
incorrect_count = len(incorrect_df)
total_count = len(df)

print(f"Total examples: {total_count}")
print(f"Correct answers: {correct_count} ({correct_count/total_count*100:.2f}%)")
print(f"Incorrect answers: {incorrect_count} ({incorrect_count/total_count*100:.2f}%)")


# In[ ]:


# Mean confidence scores for correct and incorrect answers
mean_sc_correct = correct_df['self_certainty'].mean()
mean_sc_incorrect = incorrect_df['self_certainty'].mean()

print(f"Mean self-certainty for correct answers: {mean_sc_correct:.4f}")
print(f"Mean self-certainty for incorrect answers: {mean_sc_incorrect:.4f}")


# ## Visualizations

# In[ ]:


# Plot 1: Bar chart of correct vs incorrect answers
plt.figure(figsize=(10, 6))
counts = [correct_count, incorrect_count]
labels = ['Correct', 'Incorrect']
colors = ['#2ecc71', '#e74c3c']

plt.bar(labels, counts, color=colors)
plt.title('Number of Correct vs Incorrect Answers')
plt.ylabel('Count')

# Add count and percentage labels on top of bars
for i, count in enumerate(counts):
    percentage = count / total_count * 100
    plt.text(i, count + 0.5, f"{count} ({percentage:.1f}%)", ha='center')

# plt.show()
plt.savefig('correct_incorrect.png')


# In[ ]:


# Plot 2: Comparison of mean self-certainty for correct vs incorrect answers
plt.figure(figsize=(10, 6))
sc_means = [mean_sc_correct, mean_sc_incorrect]

plt.bar(labels, sc_means, color=colors)
plt.title('Mean Self-Certainty: Correct vs Incorrect Answers')
plt.ylabel('Mean Self-Certainty Score')

# Add value labels on top of bars
for i, mean in enumerate(sc_means):
    plt.text(i, mean + 0.01, f"{mean:.4f}", ha='center')

# plt.show()
plt.savefig('correct_incorrect_sc.png')


# In[ ]:


# Plot 3: Distribution of self-certainty scores for correct vs incorrect answers
plt.figure(figsize=(12, 6))

# Create a combined dataframe with a 'correctness' column
correct_df_subset = correct_df[['self_certainty']].copy()
correct_df_subset['correctness'] = 'Correct'
incorrect_df_subset = incorrect_df[['self_certainty']].copy()
incorrect_df_subset['correctness'] = 'Incorrect'
combined_df = pd.concat([correct_df_subset, incorrect_df_subset])

# Plot the distributions
sns.histplot(data=combined_df, x='self_certainty', hue='correctness', 
             element='step', stat='density', common_norm=False, bins=20)

plt.title('Distribution of Self-Certainty Scores')
plt.xlabel('Mean Self-Certainty Score')
plt.ylabel('Density')
plt.legend(title='Answer Type')

# plt.show()
plt.savefig('correct_incorrect_sc_dist.png')


# ## Finding High-Confidence Wrong Answers

# In[ ]:


# Find incorrect answers with high confidence
# Define high confidence threshold (e.g., above the mean of correct answers)
high_confidence_threshold = mean_sc_correct

high_conf_wrong = incorrect_df[incorrect_df['self_certainty'] >= high_confidence_threshold]

print(f"Number of high-confidence wrong answers: {len(high_conf_wrong)} out of {incorrect_count} wrong answers")
print(f"Percentage: {len(high_conf_wrong)/incorrect_count*100:.2f}% of wrong answers")
print(f"Percentage of all answers: {len(high_conf_wrong)/total_count*100:.2f}%")


# In[ ]:


# Display the top 5 most confident wrong answers
if len(high_conf_wrong) > 0:
    top_confident_wrong = high_conf_wrong.sort_values('self_certainty', ascending=False).head(5)

    for i, row in top_confident_wrong.iterrows():
        print(f"Question Idx: {df.index[i]}")
        print(f"Self-Certainty: {row['self_certainty']:.4f}")
        print(f"Prompt: {row['full_prompt'][:100]}...")
        print(f"Completion: {row['predictions']}")
        print("-" * 80)


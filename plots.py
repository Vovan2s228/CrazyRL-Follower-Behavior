import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('evaluation_log_follower_behavior.csv')

# Count predicted behaviors
behavior_counts = df['PredictedBehavior'].value_counts()
print("Predicted Behavior Counts:\n", behavior_counts)

# Bar plot of predicted behavior counts
behavior_counts.plot(kind='bar', title='Predicted Behavior Distribution')
plt.xlabel('Behavior')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('C:/Users/kalin/Downloads/predicted_behavior_distribution.png')

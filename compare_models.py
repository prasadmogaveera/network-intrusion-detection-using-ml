
import pandas as pd
import matplotlib.pyplot as plt

# Previously recorded evaluation results
results = [
    {
        "Model": "Random Forest",
        "Accuracy": 0.9986,
        "Anomaly_Precision": 0.9991,
        "Anomaly_Recall": 0.9994,
        "Anomaly_F1": 0.9992
    },
    {
        "Model": "XGBoost",
        "Accuracy": 0.9984,
        "Anomaly_Precision": 0.9984,
        "Anomaly_Recall": 0.9999,
        "Anomaly_F1": 0.9991
    },
    {
        "Model": "ANN",
        "Accuracy": 0.9913,
        "Anomaly_Precision": 0.9966,
        "Anomaly_Recall": 0.9939,
        "Anomaly_F1": 0.9952
    },
    {
        "Model": "LSTM",
        "Accuracy": 0.9949012109623965,
        "Anomaly_Precision": 0.9971704786606932,
        "Anomaly_Recall": 0.9971704786606932,
        "Anomaly_F1": 0.9971704786606932
    }
]

df = pd.DataFrame(results)

# Save the comparison table
df.to_csv("model_comparison.csv", index=False)

# Display percentages in the terminal
display_df = df.copy()
metric_columns = [
    "Accuracy",
    "Anomaly_Precision",
    "Anomaly_Recall",
    "Anomaly_F1"
]

display_df[metric_columns] = (
    display_df[metric_columns] * 100
).round(2)

print("\nMODEL PERFORMANCE COMPARISON")
print(display_df.to_string(index=False))

# Create a comparison chart
display_df.set_index("Model")[metric_columns].plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("IoT Intrusion Detection - Model Comparison")
plt.ylabel("Score (%)")
plt.xlabel("Model")
plt.ylim(90, 101)
plt.xticks(rotation=0)
plt.legend([
    "Accuracy",
    "Anomaly Precision",
    "Anomaly Recall",
    "Anomaly F1-score"
])
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.show()

print("\nSaved: model_comparison.csv")
print("Saved: model_comparison.png")

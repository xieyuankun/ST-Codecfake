import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

output_file = 'result/result_id.txt'
label_file = "/data7/xyk/codecfake_st/v2/label/eval_id.txt"      



with open(output_file, 'r') as f:
    output_lines = f.read().splitlines()

with open(label_file, 'r') as f:
    label_lines = f.read().splitlines()

# Extract true labels and predicted labels
true_labels = [int(line.split()[2]) for line in label_lines]
predicted_labels = [int(line.split()[1]) for line in output_lines]

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Calculate the overall F1 score
f1_total = f1_score(true_labels, predicted_labels, average='macro')

# Calculate F1 scores for each class
f1_per_class = f1_score(true_labels, predicted_labels, average=None)

# Print results
print(f"Overall F1 score: {f1_total:.4f}")
for class_id, f1 in enumerate(f1_per_class):
    print(f"F1 score - Class {class_id}: {f1:.4f}")
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import numpy as np



output_file = './result/result_C4-3.txt'


with open(output_file, 'r') as f:
    output_lines = f.read().splitlines()

# Extract true labels and predicted labels
predicted_labels = [int(line.split()[1]) for line in output_lines]
true_labels = [4] * len(predicted_labels)


f1_micro = f1_score(true_labels, predicted_labels, average='micro')


print(f"F1 Score (Micro): {f1_micro:.4f}")



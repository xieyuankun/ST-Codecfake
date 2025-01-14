import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import numpy as np


# output_file = 'result/result_OOD_7.txt'
# label_file = "/data7/xyk/codecfake_st/label/OOD_7.txt"    

output_file = '/data3/xyk/codecfake_st/st_codecfake_benchmark/result/result_unseenreal_ncssd.txt'

with open(output_file, 'r') as f:
    output_lines = f.read().splitlines()

# Extract true labels and predicted labels
predicted_labels = [int(line.split()[1]) for line in output_lines]
true_labels = [0] * len(predicted_labels)
print(len(true_labels))

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

acc = accuracy_score(true_labels, predicted_labels)
per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

# print(f"Accuracy all: {acc:.4f}")
# for class_id, acc in enumerate(per_class_acc):
#     print(f"Accuracy - Class {class_id}: {acc:.4f}")


f1_micro = f1_score(true_labels, predicted_labels, average='micro')


print(f"F1 Score (Micro): {f1_micro:.4f}")



from sklearn.metrics import f1_score

def calculate_f1_score(output_file, true_label):
    with open(output_file, 'r') as f:
        output_lines = f.read().splitlines()
    predicted_labels = [int(line.split()[1]) for line in output_lines]
    true_labels = [true_label] * len(predicted_labels)
    f1_micro = f1_score(true_labels, predicted_labels, average='micro')
    print(f"F1 Score (Micro) for {output_file}: {f1_micro:.4f}")

# Define the output files and their corresponding true labels
output_files = {
    './result/alm/moshi.txt': 1,
    './result/alm/speechgpt.txt': 2,
    './result/alm/valle.txt': 4,
    './result/alm/miniomni.txt': 6
}

# Calculate and print F1 scores for each output file
for output_file, true_label in output_files.items():
    calculate_f1_score(output_file, true_label)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load all runs (sheets)
file_path = 'results_20250501_102133.xlsx'  # <-- replace with your file path
sheet_names = [f'Run {i}' for i in range(1, 10)]  # Run 1 to Run 9

# Read all sheets
dfs = [pd.read_excel(file_path, sheet_name=sheet) for sheet in sheet_names]

datasets = ['pavia', 'Botswana', 'indian_pines_corrected', 'KSC', 'salinas_corrected']
models = ['LR', 'MLP', 'CNN', 'TNN']
sample_sizes = [5, 10, 25, 50]

# Prepare a dictionary to store results
avg_results = {dataset: {sample: {model: [] for model in models} for sample in sample_sizes} for dataset in datasets}

# Collect accuracies from each run and dataset
for dataset in datasets:
    for sample_size in sample_sizes:
        # Collect accuracy values for each model from all sheets
        lr_values, mlp_values, cnn_values, tnn_values = [], [], [], []
                                       
        for df in dfs:
            # --- Force 'Samples' column to integers ---
            df['Samples'] = df['Samples'].astype(int)

            # Filter rows matching the dataset and sample size
            subset = df[(df['Dataset'] == dataset) & (df['Samples'] == sample_size)]
            for df_idx, df in enumerate(dfs):
                # Force 'Samples' to integers
                df['Samples'] = df['Samples'].astype(int)

                subset = df[(df['Dataset'] == dataset) & (df['Samples'] == sample_size)]

            if not subset.empty:
                # Append the accuracy values for each model
                lr_values.append(subset['LR'].values[0])
                mlp_values.append(subset['MLP'].values[0])
                cnn_values.append(subset['CNN'].values[0])
                tnn_values.append(subset['TNN'].values[0])

        # Store the accuracies for final averaging
        avg_results[dataset][sample_size]['LR'] = lr_values
        avg_results[dataset][sample_size]['MLP'] = mlp_values
        avg_results[dataset][sample_size]['CNN'] = cnn_values
        avg_results[dataset][sample_size]['TNN'] = tnn_values

# Compute the final averages
final_avg = {dataset: {model: [] for model in models} for dataset in datasets}

for dataset in datasets:
    for model in models:
        for sample in sample_sizes:
            acc_list = avg_results[dataset][sample][model]
            if acc_list:
                avg = sum(acc_list) / len(acc_list)
            else:
                avg = None
            final_avg[dataset][model].append(avg)

save_dir = 'plots'
os.makedirs(save_dir, exist_ok=True)

for dataset in datasets:
    plt.figure(figsize=(10,6))
    bar_width = 0.2
    x = np.arange(len(sample_sizes))

    for idx, model in enumerate(models):
        plt.bar(x + idx*bar_width, final_avg[dataset][model], width=bar_width, label=model)
    
    plt.title(f'Accuracy vs Samples for {dataset.capitalize()} Dataset')
    plt.xlabel('Number of Samples')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(x + bar_width*(len(models)-1)/2, sample_sizes)
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(save_dir, f'{dataset}_accuracy_vs_samples.png')
    plt.savefig(save_path)

    plt.close()  # Close the figure after saving
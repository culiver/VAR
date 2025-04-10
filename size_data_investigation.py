import os
import os.path as osp
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd

# Define paths
DATA_DIR = "./model_size_analysis/imagenet10/var/var_cfg[0]_dist_prob_backup"
OUTPUT_DIR = "./model_size_analysis/imagenet10/var/performance_gap_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ImageNet 10 class names for better visualization
CLASS_NAMES = [
    "tench", "goldfish", "great white shark", "tiger shark", 
    "hammerhead shark", "electric ray", "stingray", "cock", 
    "hen", "ostrich"
]

def load_json_files(data_dir):
    """Load all JSON result files from the directory."""
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json') and f[0].isdigit()]
    data = []
    
    for json_file in json_files:
        with open(osp.join(data_dir, json_file), 'r') as f:
            result = json.load(f)
            # Add the file name (sample_id) to the result
            result['sample_id'] = int(json_file.split('.')[0])
            data.append(result)
    
    return data

def analyze_model_differences(data):
    """Analyze differences between d16 and d30 model predictions."""
    # Categories based on model performance
    d16_correct_d30_wrong = []
    d16_wrong_d30_correct = []
    both_correct = []
    both_wrong = []
    
    for item in data:
        if item['pred_d16'] == item['label'] and item['pred_d30'] != item['label']:
            d16_correct_d30_wrong.append(item)
        elif item['pred_d16'] != item['label'] and item['pred_d30'] == item['label']:
            d16_wrong_d30_correct.append(item)
        elif item['pred_d16'] == item['label'] and item['pred_d30'] == item['label']:
            both_correct.append(item)
        else:
            both_wrong.append(item)
    
    # Print summary statistics
    total = len(data)
    print(f"Total samples: {total}")
    print(f"Both models correct: {len(both_correct)} ({len(both_correct)/total*100:.2f}%)")
    print(f"Both models wrong: {len(both_wrong)} ({len(both_wrong)/total*100:.2f}%)")
    print(f"D16 correct, D30 wrong: {len(d16_correct_d30_wrong)} ({len(d16_correct_d30_wrong)/total*100:.2f}%)")
    print(f"D16 wrong, D30 correct: {len(d16_wrong_d30_correct)} ({len(d16_wrong_d30_correct)/total*100:.2f}%)")
    
    # Calculate accuracy for each model
    d16_accuracy = (len(both_correct) + len(d16_correct_d30_wrong)) / total
    d30_accuracy = (len(both_correct) + len(d16_wrong_d30_correct)) / total
    print(f"D16 accuracy: {d16_accuracy*100:.2f}%")
    print(f"D30 accuracy: {d30_accuracy*100:.2f}%")
    
    return {
        'd16_correct_d30_wrong': d16_correct_d30_wrong,
        'd16_wrong_d30_correct': d16_wrong_d30_correct,
        'both_correct': both_correct,
        'both_wrong': both_wrong
    }

def analyze_confusion_patterns(category_data):
    """Analyze confusion patterns where d16 is correct but d30 is wrong."""
    d16_correct_d30_wrong = category_data['d16_correct_d30_wrong']
    
    # Create confusion matrix
    confusion_data = defaultdict(int)
    for item in d16_correct_d30_wrong:
        true_class = item['label']
        d30_pred = item['pred_d30']
        confusion_data[(true_class, d30_pred)] += 1
    
    # Convert to matrix format
    classes = sorted(set([x[0] for x in confusion_data.keys()] + [x[1] for x in confusion_data.keys()]))
    confusion_matrix = np.zeros((len(classes), len(classes)))
    
    for (true_class, pred_class), count in confusion_data.items():
        true_idx = classes.index(true_class)
        pred_idx = classes.index(pred_class)
        confusion_matrix[true_idx, pred_idx] = count
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=[CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else f"Class {c}" for c in classes],
                yticklabels=[CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else f"Class {c}" for c in classes])
    plt.xlabel('D30 Prediction')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix for Cases where D16 is Correct but D30 is Wrong')
    plt.tight_layout()
    plt.savefig(osp.join(OUTPUT_DIR, 'd16_correct_d30_wrong_confusion.png'))
    
    return confusion_matrix, classes


def analyze_specific_samples(df_likelihood, n=5):
    """Select and analyze specific interesting samples for deeper investigation."""
    # Sort by largest likelihood difference
    interesting_by_target = df_likelihood.sort_values('d16_d30_target_diff', ascending=False).head(n)
    interesting_by_margin = df_likelihood.sort_values('d30_margin', ascending=True).head(n)
    
    # Combine interesting samples
    all_interesting = pd.concat([interesting_by_target, interesting_by_margin]).drop_duplicates()
    
    # Print details for these samples
    print("\nInteresting samples for further investigation:")
    for i, row in all_interesting.iterrows():
        # Convert class labels to integers for indexing CLASS_NAMES
        label_int = int(row['label'])
        d30_pred_int = int(row['d30_pred'])
        
        print(f"Sample ID: {row['sample_id']}")
        print(f"  True class: {label_int} ({CLASS_NAMES[label_int] if label_int < len(CLASS_NAMES) else 'Unknown'})")
        print(f"  D30 predicted: {d30_pred_int} ({CLASS_NAMES[d30_pred_int] if d30_pred_int < len(CLASS_NAMES) else 'Unknown'})")
        print(f"  D16 target log likelihood: {row['d16_target_ll']:.2f}")
        print(f"  D30 target log likelihood: {row['d30_target_ll']:.2f}")
        print(f"  Difference: {row['d16_d30_target_diff']:.2f}")
        print(f"  D16 margin: {row['d16_margin']:.2f}")
        print(f"  D30 margin: {row['d30_margin']:.2f}")
        print()
    
    return all_interesting

def analyze_layer_differences(category_data, output_dir):
    """Analyze how layer-specific performance differs between d16 and d30 models."""
    # Get the data where d16 is correct but d30 is wrong
    d16_correct_d30_wrong = category_data['d16_correct_d30_wrong']
    all_data = category_data['d16_correct_d30_wrong'] + category_data['d16_wrong_d30_correct'] + category_data['both_correct'] + category_data['both_wrong']
    
    # Define paths to layer analysis directories
    layer_acc_analysis_dir = osp.join(DATA_DIR, "layer_acc_analysis")
    
    # Collect data at each scale (layer)
    scales = list(range(10))  # Assuming 10 scales (0-9)
    
    # Initialize data structures to store accumulated layer analysis results
    d16_correct_d30_wrong_ids = [item['sample_id'] for item in d16_correct_d30_wrong]
    all_sample_ids = [item['sample_id'] for item in all_data]
    
    # Store accumulated layer data for both sample sets
    focused_layer_data = {scale: [] for scale in scales}  # For d16_correct_d30_wrong cases
    all_layer_data = {scale: [] for scale in scales}      # For all cases
    
    # Process accumulated data for each sample and each scale
    # First for d16_correct_d30_wrong cases
    for sample_id in d16_correct_d30_wrong_ids:
        for scale in scales:
            acc_layer_file = osp.join(layer_acc_analysis_dir, f"{sample_id}_{scale}-layer_acc.json")
            if osp.exists(acc_layer_file):
                with open(acc_layer_file, 'r') as f:
                    acc_layer_data = json.load(f)
                    acc_layer_data['sample_id'] = sample_id
                    acc_layer_data['scale'] = scale
                    focused_layer_data[scale].append(acc_layer_data)
    
    # Then for all cases
    for sample_id in all_sample_ids:
        for scale in scales:
            acc_layer_file = osp.join(layer_acc_analysis_dir, f"{sample_id}_{scale}-layer_acc.json")
            if osp.exists(acc_layer_file):
                with open(acc_layer_file, 'r') as f:
                    acc_layer_data = json.load(f)
                    acc_layer_data['sample_id'] = sample_id
                    acc_layer_data['scale'] = scale
                    all_layer_data[scale].append(acc_layer_data)
    
    # Analyze accuracy at each accumulated scale for d16_correct_d30_wrong cases
    focused_accumulated_data = []
    for scale in scales:
        if not focused_layer_data[scale]:
            continue
        
        # Count cases where d16 is correct and d30 is incorrect at this accumulated scale
        d16_correct = sum(1 for data in focused_layer_data[scale] if data['pred_d16'] == data['label'])
        d30_correct = sum(1 for data in focused_layer_data[scale] if data['pred_d30'] == data['label'])
        total = len(focused_layer_data[scale])
        
        if total > 0:
            focused_accumulated_data.append({
                'scale': scale,
                'patch_size': 2**scale,
                'd16_acc': d16_correct / total * 100,
                'd30_acc': d30_correct / total * 100,
                'acc_diff': (d16_correct - d30_correct) / total * 100,
                'sample_count': total
            })
    
    # Analyze accuracy at each accumulated scale for all cases
    all_accumulated_data = []
    for scale in scales:
        if not all_layer_data[scale]:
            continue
        
        # Count correct predictions for each model
        d16_correct = sum(1 for data in all_layer_data[scale] if data['pred_d16'] == data['label'])
        d30_correct = sum(1 for data in all_layer_data[scale] if data['pred_d30'] == data['label'])
        total = len(all_layer_data[scale])
        
        if total > 0:
            all_accumulated_data.append({
                'scale': scale,
                'patch_size': 2**scale,
                'd16_acc': d16_correct / total * 100,
                'd30_acc': d30_correct / total * 100,
                'acc_diff': (d16_correct - d30_correct) / total * 100,
                'sample_count': total
            })
    
    # Convert to DataFrames
    focused_acc_df = pd.DataFrame(focused_accumulated_data)
    all_acc_df = pd.DataFrame(all_accumulated_data)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Model accuracy at each accumulated scale (d16 correct d30 wrong cases)
    if not focused_acc_df.empty:
        sns.lineplot(x='scale', y='d16_acc', data=focused_acc_df, marker='o', label='D16 Accuracy', ax=ax1)
        sns.lineplot(x='scale', y='d30_acc', data=focused_acc_df, marker='s', label='D30 Accuracy', ax=ax1)
        ax1.set_title('Model Accuracy by Scale\n(D16 Correct, D30 Wrong Cases)')
        ax1.set_xlabel('Scale Index (Patch Size = 2^scale)')
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='lower right')
        
        # Annotate the count of samples at each scale
        for idx, row in focused_acc_df.iterrows():
            ax1.annotate(f"n={int(row['sample_count'])}", 
                        (row['scale'], min(row['d16_acc'], row['d30_acc'])-5),
                        fontsize=8)
    
    # Plot 2: Model accuracy at each accumulated scale (all cases)
    if not all_acc_df.empty:
        sns.lineplot(x='scale', y='d16_acc', data=all_acc_df, marker='o', label='D16 Accuracy', ax=ax2)
        sns.lineplot(x='scale', y='d30_acc', data=all_acc_df, marker='s', label='D30 Accuracy', ax=ax2)
        ax2.set_title('Model Accuracy by Scale\n(All Data)')
        ax2.set_xlabel('Scale Index (Patch Size = 2^scale)')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='lower right')
        
        # Annotate the count of samples at each scale
        for idx, row in all_acc_df.iterrows():
            ax2.annotate(f"n={int(row['sample_count'])}", 
                        (row['scale'], min(row['d16_acc'], row['d30_acc'])-5),
                        fontsize=8)
    
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, 'model_accuracy_by_scale.png'))
    
    # Print a summary of accuracy by scale
    print("\nAccuracy by Scale Summary:")
    
    if not focused_acc_df.empty:
        print("\nFor D16 Correct, D30 Wrong cases:")
        for i, row in focused_acc_df.iterrows():
            print(f"  Scale {int(row['scale'])} (Patch Size {int(row['patch_size'])}):")
            print(f"    D16 accuracy: {row['d16_acc']:.2f}%")
            print(f"    D30 accuracy: {row['d30_acc']:.2f}%")
            print(f"    Difference (D16-D30): {row['acc_diff']:.2f}%")
            print(f"    Sample count: {int(row['sample_count'])}")
    
    if not all_acc_df.empty:
        print("\nFor All Data:")
        for i, row in all_acc_df.iterrows():
            print(f"  Scale {int(row['scale'])} (Patch Size {int(row['patch_size'])}):")
            print(f"    D16 accuracy: {row['d16_acc']:.2f}%")
            print(f"    D30 accuracy: {row['d30_acc']:.2f}%")
            print(f"    Difference (D16-D30): {row['acc_diff']:.2f}%")
            print(f"    Sample count: {int(row['sample_count'])}")
    
    return {
        'focused_accumulated': focused_acc_df,
        'all_accumulated': all_acc_df
    }

def compare_var_l2dist_methods(output_dir):
    """Compare VAR and L2_dist methods for both layerwise and layer_acc data."""
    # Define paths to data directories
    var_layer_acc_dir = "./model_size_analysis/imagenet10/var/var_cfg[0.0]/layer_acc"
    var_layerwise_dir = "./model_size_analysis/imagenet10/var/var_cfg[0.0]/layerwise"
    l2dist_layer_acc_dir = "./model_size_analysis/imagenet10/l2_dist/var_cfg[0.0]/layer_acc"
    l2dist_layerwise_dir = "./model_size_analysis/imagenet10/l2_dist/var_cfg[0.0]/layerwise"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data for each scale (0-9)
    scales = list(range(10))
    
    # Data structures to store results
    data = {
        'var_layer_acc': {scale: [] for scale in scales},
        'var_layerwise': {scale: [] for scale in scales},
        'l2dist_layer_acc': {scale: [] for scale in scales},
        'l2dist_layerwise': {scale: [] for scale in scales}
    }
    
    # Load and process data from all directories
    for scale in scales:
        # Process VAR layer_acc files
        for file in os.listdir(var_layer_acc_dir):
            if file.endswith(f"{scale}-layer_acc.json"):
                with open(osp.join(var_layer_acc_dir, file), 'r') as f:
                    try:
                        item = json.load(f)
                        item['sample_id'] = int(file.split('_')[0])
                        item['scale'] = scale
                        data['var_layer_acc'][scale].append(item)
                    except json.JSONDecodeError:
                        print(f"Error reading {file} in var_layer_acc_dir")
        
        # Process VAR layerwise files
        for file in os.listdir(var_layerwise_dir):
            if file.endswith(f"{scale}-layer.json"):
                with open(osp.join(var_layerwise_dir, file), 'r') as f:
                    try:
                        item = json.load(f)
                        item['sample_id'] = int(file.split('_')[0])
                        item['scale'] = scale
                        data['var_layerwise'][scale].append(item)
                    except json.JSONDecodeError:
                        print(f"Error reading {file} in var_layerwise_dir")
        
        # Process L2_dist layer_acc files
        for file in os.listdir(l2dist_layer_acc_dir):
            if file.endswith(f"{scale}-layer_acc.json"):
                with open(osp.join(l2dist_layer_acc_dir, file), 'r') as f:
                    try:
                        item = json.load(f)
                        item['sample_id'] = int(file.split('_')[0])
                        item['scale'] = scale
                        data['l2dist_layer_acc'][scale].append(item)
                    except json.JSONDecodeError:
                        print(f"Error reading {file} in l2dist_layer_acc_dir")
        
        # Process L2_dist layerwise files
        for file in os.listdir(l2dist_layerwise_dir):
            if file.endswith(f"{scale}-layer.json"):
                with open(osp.join(l2dist_layerwise_dir, file), 'r') as f:
                    try:
                        item = json.load(f)
                        item['sample_id'] = int(file.split('_')[0])
                        item['scale'] = scale
                        data['l2dist_layerwise'][scale].append(item)
                    except json.JSONDecodeError:
                        print(f"Error reading {file} in l2dist_layerwise_dir")
    
    # Analyze accuracy at each scale for both methods
    summary = {
        'var_layer_acc': [],
        'var_layerwise': [],
        'l2dist_layer_acc': [],
        'l2dist_layerwise': []
    }
    
    # Process layerwise data (individual scale contributions)
    for scale in scales:
        for method_key in ['var_layerwise', 'l2dist_layerwise']:
            if not data[method_key][scale]:
                continue
            
            # Count correct predictions for each model
            items = data[method_key][scale]
            d16_correct = sum(1 for item in items if item['pred_d16'] == item['label'])
            d30_correct = sum(1 for item in items if item['pred_d30'] == item['label'])
            total = len(items)
            
            if total > 0:
                summary[method_key].append({
                    'scale': scale,
                    'patch_size': 2**scale,
                    'd16_acc': d16_correct / total * 100,
                    'd30_acc': d30_correct / total * 100,
                    'acc_diff': (d16_correct - d30_correct) / total * 100,
                    'sample_count': total
                })
    
    # Process layer_acc data (accumulated up to each scale)
    for scale in scales:
        for method_key in ['var_layer_acc', 'l2dist_layer_acc']:
            if not data[method_key][scale]:
                continue
            
            # Count correct predictions for each model
            items = data[method_key][scale]
            d16_correct = sum(1 for item in items if item['pred_d16'] == item['label'])
            d30_correct = sum(1 for item in items if item['pred_d30'] == item['label'])
            total = len(items)
            
            if total > 0:
                summary[method_key].append({
                    'scale': scale,
                    'patch_size': 2**scale,
                    'd16_acc': d16_correct / total * 100,
                    'd30_acc': d30_correct / total * 100,
                    'acc_diff': (d16_correct - d30_correct) / total * 100,
                    'sample_count': total
                })
    
    # Convert to DataFrames
    dfs = {}
    for key in summary:
        if summary[key]:
            dfs[key] = pd.DataFrame(summary[key])
        else:
            dfs[key] = pd.DataFrame()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Layerwise comparison (individual scale contributions)
    ax1.set_title('Layerwise Accuracy Comparison\n(Individual Scale Contributions)', fontsize=14)
    ax1.set_xlabel('Scale Index (Patch Size = 2^scale)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add var layerwise data to plot
    if not dfs['var_layerwise'].empty:
        sns.lineplot(x='scale', y='d16_acc', data=dfs['var_layerwise'], 
                     marker='o', label='VAR D16', color='blue', ax=ax1)
        sns.lineplot(x='scale', y='d30_acc', data=dfs['var_layerwise'], 
                     marker='s', label='VAR D30', color='lightblue', ax=ax1)
        # Annotate VAR sample count for reference
        for idx, row in dfs['var_layerwise'].iterrows():
            ax1.annotate(f"n={int(row['sample_count'])}", 
                        (row['scale'], row['d16_acc']-3),
                        fontsize=8, color='blue', alpha=0.7)
    
    # Add l2dist layerwise data to plot
    if not dfs['l2dist_layerwise'].empty:
        sns.lineplot(x='scale', y='d16_acc', data=dfs['l2dist_layerwise'], 
                     marker='^', label='L2_DIST D16', color='red', ax=ax1)
        sns.lineplot(x='scale', y='d30_acc', data=dfs['l2dist_layerwise'], 
                     marker='d', label='L2_DIST D30', color='lightcoral', ax=ax1)
    
    ax1.legend(loc='lower right')
    
    # Plot 2: Layer_acc comparison (accumulated up to each scale)
    ax2.set_title('Accumulated Accuracy Comparison\n(Up to Each Scale)', fontsize=14)
    ax2.set_xlabel('Scale Index (Patch Size = 2^scale)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add var layer_acc data to plot
    if not dfs['var_layer_acc'].empty:
        sns.lineplot(x='scale', y='d16_acc', data=dfs['var_layer_acc'], 
                     marker='o', label='VAR D16', color='blue', ax=ax2)
        sns.lineplot(x='scale', y='d30_acc', data=dfs['var_layer_acc'], 
                     marker='s', label='VAR D30', color='lightblue', ax=ax2)
        # Annotate VAR sample count for reference
        for idx, row in dfs['var_layer_acc'].iterrows():
            ax2.annotate(f"n={int(row['sample_count'])}", 
                        (row['scale'], row['d16_acc']-3),
                        fontsize=8, color='blue', alpha=0.7)
    
    # Add l2dist layer_acc data to plot
    if not dfs['l2dist_layer_acc'].empty:
        sns.lineplot(x='scale', y='d16_acc', data=dfs['l2dist_layer_acc'], 
                     marker='^', label='L2_DIST D16', color='red', ax=ax2)
        sns.lineplot(x='scale', y='d30_acc', data=dfs['l2dist_layer_acc'], 
                     marker='d', label='L2_DIST D30', color='lightcoral', ax=ax2)
    
    ax2.legend(loc='lower right')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(osp.join(output_dir, 'var_vs_l2dist_comparison.png'))
    
    # Print summary
    print("\nComparison of VAR and L2_DIST Methods:")
    
    print("\nLayerwise (Individual Scale Contributions):")
    print("------------------------------------------")
    print("VAR Method:")
    for idx, row in dfs['var_layerwise'].iterrows():
        scale = int(row['scale'])
        print(f"  Scale {scale} (Patch Size {int(row['patch_size'])}):")
        print(f"    D16 accuracy: {row['d16_acc']:.2f}%")
        print(f"    D30 accuracy: {row['d30_acc']:.2f}%")
        print(f"    Difference (D16-D30): {row['acc_diff']:.2f}%")
        print(f"    Sample count: {int(row['sample_count'])}")
    
    print("\nL2_DIST Method:")
    for idx, row in dfs['l2dist_layerwise'].iterrows():
        scale = int(row['scale'])
        print(f"  Scale {scale} (Patch Size {int(row['patch_size'])}):")
        print(f"    D16 accuracy: {row['d16_acc']:.2f}%")
        print(f"    D30 accuracy: {row['d30_acc']:.2f}%")
        print(f"    Difference (D16-D30): {row['acc_diff']:.2f}%")
        print(f"    Sample count: {int(row['sample_count'])}")
    
    print("\nLayer_acc (Accumulated Up to Each Scale):")
    print("----------------------------------------")
    print("VAR Method:")
    for idx, row in dfs['var_layer_acc'].iterrows():
        scale = int(row['scale'])
        print(f"  Scale {scale} (Patch Size {int(row['patch_size'])}):")
        print(f"    D16 accuracy: {row['d16_acc']:.2f}%")
        print(f"    D30 accuracy: {row['d30_acc']:.2f}%")
        print(f"    Difference (D16-D30): {row['acc_diff']:.2f}%")
        print(f"    Sample count: {int(row['sample_count'])}")
    
    print("\nL2_DIST Method:")
    for idx, row in dfs['l2dist_layer_acc'].iterrows():
        scale = int(row['scale'])
        print(f"  Scale {scale} (Patch Size {int(row['patch_size'])}):")
        print(f"    D16 accuracy: {row['d16_acc']:.2f}%")
        print(f"    D30 accuracy: {row['d30_acc']:.2f}%")
        print(f"    Difference (D16-D30): {row['acc_diff']:.2f}%")
        print(f"    Sample count: {int(row['sample_count'])}")
    
    return dfs

def main():
    # Load data
    print("Loading data from", DATA_DIR)
    data = load_json_files(DATA_DIR)
    print(f"Loaded {len(data)} samples")
    
    # Basic model comparison
    print("\n" + "="*50)
    print("ANALYZING MODEL PERFORMANCE DIFFERENCES")
    print("="*50)
    category_data = analyze_model_differences(data)
    
    # Comparison between VAR and L2_dist methods
    print("\n" + "="*50)
    print("COMPARING VAR AND L2_DIST METHODS")
    print("="*50)
    method_comparison = compare_var_l2dist_methods(OUTPUT_DIR)
    
    # Other analyses can be uncommented as needed
    # confusion_matrix, classes = analyze_confusion_patterns(category_data)
    # layer_dfs = analyze_layer_differences(category_data, OUTPUT_DIR)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

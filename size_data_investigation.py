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

def analyze_likelihood_differences(category_data):
    """Analyze log likelihood differences in the focus category (d16 correct, d30 wrong)."""
    d16_correct_d30_wrong = category_data['d16_correct_d30_wrong']
    
    # Extract likelihood data
    likelihood_data = []
    for item in d16_correct_d30_wrong:
        label = item['label']
        d16_target_ll = item['target_log_likelihood_d16']
        d30_target_ll = item['target_log_likelihood_d30']
        
        # Find what d30 predicted instead
        d30_pred = item['pred_d30']
        
        # Find likelihood of d30's prediction in both models
        d16_ll_list = item['log_likelihood_d16'][:-1]  # Exclude unconditional
        d30_ll_list = item['log_likelihood_d30'][:-1]  # Exclude unconditional
        
        d16_d30pred_ll = d16_ll_list[d30_pred]
        d30_d30pred_ll = d30_ll_list[d30_pred]
        
        # Calculate margins and differences
        d16_margin = d16_target_ll - d16_d30pred_ll
        d30_margin = d30_target_ll - d30_d30pred_ll
        
        likelihood_data.append({
            'sample_id': item['sample_id'],
            'label': label,
            'd30_pred': d30_pred,
            'd16_target_ll': d16_target_ll,
            'd30_target_ll': d30_target_ll,
            'd16_d30pred_ll': d16_d30pred_ll,
            'd30_d30pred_ll': d30_d30pred_ll,
            'd16_margin': d16_margin,
            'd30_margin': d30_margin,
            'd16_d30_target_diff': d16_target_ll - d30_target_ll,
            'd16_d30_wrong_diff': d16_d30pred_ll - d30_d30pred_ll
        })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(likelihood_data)
    
    # Analyze and plot the differences
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    sns.histplot(df['d16_d30_target_diff'], kde=True)
    plt.title('D16 - D30 Log Likelihood Difference for True Class')
    plt.xlabel('Difference')
    plt.axvline(x=0, color='r', linestyle='--')
    
    plt.subplot(2, 2, 2)
    sns.histplot(df['d16_margin'], color='blue', kde=True, label='D16')
    sns.histplot(df['d30_margin'], color='red', kde=True, label='D30')
    plt.title('Margin between True Class and D30\'s Prediction')
    plt.xlabel('Margin (True - D30 Pred)')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    error_classes = df['d30_pred'].value_counts().reset_index()
    error_classes.columns = ['class', 'count']
    sns.barplot(x='class', y='count', data=error_classes)
    plt.title('D30 Error Distribution by Predicted Class')
    plt.xlabel('Class ID')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    ground_truth_classes = df['label'].value_counts().reset_index()
    ground_truth_classes.columns = ['class', 'count']
    sns.barplot(x='class', y='count', data=ground_truth_classes)
    plt.title('D30 Error Distribution by Ground Truth Class')
    plt.xlabel('Class ID')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(osp.join(OUTPUT_DIR, 'd16_correct_d30_wrong_analysis.png'))
    
    return df

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

def analyze_log_likelihood_patterns(category_data):
    """Analyze patterns in log likelihood distributions."""
    d16_correct_d30_wrong = category_data['d16_correct_d30_wrong']
    
    # For each sample where d16 is correct but d30 is wrong,
    # analyze the full likelihood distribution across all classes
    
    # Calculate rankings of true class in both models
    rank_data = []
    for item in d16_correct_d30_wrong:
        d16_lls = np.array(item['log_likelihood_d16'][:-1])  # Exclude unconditional
        d30_lls = np.array(item['log_likelihood_d30'][:-1])  # Exclude unconditional
        
        true_class = item['label']
        
        # Get rankings (higher log likelihood = better)
        d16_ranking = len(d16_lls) - np.argsort(np.argsort(d16_lls))[true_class]
        d30_ranking = len(d30_lls) - np.argsort(np.argsort(d30_lls))[true_class]
        
        # Calculate differences between top class and true class
        d16_top_idx = np.argmax(d16_lls)
        d30_top_idx = np.argmax(d30_lls)
        
        d16_diff_to_top = d16_lls[d16_top_idx] - d16_lls[true_class] if d16_top_idx != true_class else 0
        d30_diff_to_top = d30_lls[d30_top_idx] - d30_lls[true_class] if d30_top_idx != true_class else 0
        
        # Calculate standard deviation of log likelihoods
        d16_std = np.std(d16_lls)
        d30_std = np.std(d30_lls)
        
        rank_data.append({
            'sample_id': item['sample_id'],
            'label': true_class,
            'd16_rank': d16_ranking,
            'd30_rank': d30_ranking,
            'd16_diff_to_top': d16_diff_to_top,
            'd30_diff_to_top': d30_diff_to_top,
            'd16_std': d16_std,
            'd30_std': d30_std
        })
    
    # Convert to DataFrame
    rank_df = pd.DataFrame(rank_data)
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(rank_df['d16_rank'], color='blue', kde=True, label='D16')
    sns.histplot(rank_df['d30_rank'], color='red', kde=True, label='D30')
    plt.title('Ranking of True Class in Log Likelihood')
    plt.xlabel('Rank (1 = highest log likelihood)')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    sns.histplot(rank_df['d30_diff_to_top'], kde=True)
    plt.title('D30: Difference between Top Class and True Class')
    plt.xlabel('Log Likelihood Difference')
    
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=rank_df, x='d16_std', y='d30_std')
    plt.plot([0, plt.xlim()[1]], [0, plt.xlim()[1]], 'r--')
    plt.title('Standard Deviation of Log Likelihoods')
    plt.xlabel('D16 Std Dev')
    plt.ylabel('D30 Std Dev')
    
    plt.subplot(2, 2, 4)
    sns.histplot(rank_df['d16_std'] - rank_df['d30_std'], kde=True)
    plt.title('Difference in Std Dev (D16 - D30)')
    plt.xlabel('Std Dev Difference')
    plt.axvline(x=0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(osp.join(OUTPUT_DIR, 'rank_analysis.png'))
    
    return rank_df

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
    
    # Focus on d16 correct, d30 wrong cases
    print("\n" + "="*50)
    print("ANALYZING D16 CORRECT, D30 WRONG CASES")
    print("="*50)
    
    # Log likelihood differences analysis
    df_likelihood = analyze_likelihood_differences(category_data)
    
    # Confusion patterns
    confusion_matrix, classes = analyze_confusion_patterns(category_data)
    
    # Ranking analysis
    rank_df = analyze_log_likelihood_patterns(category_data)
    
    # Layer-specific analysis (using the layer_analysis, layer_acc_analysis, and layer_cond_analysis directories)
    print("\n" + "="*50)
    print("ANALYZING LAYER-SPECIFIC DIFFERENCES")
    print("="*50)
    layer_dfs = analyze_layer_differences(category_data, OUTPUT_DIR)
    
    # Interesting samples for further investigation
    interesting_samples = analyze_specific_samples(df_likelihood)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

# scripts/data/generate_partitions.py
import json
import os
import random
import argparse
import yaml
from pathlib import Path
import math
import numpy as np
from skmultilearn.model_selection import IterativeStratification

def read_jsonl(file_path):
    """
    Reads a .jsonl file and returns a list of dictionaries.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """
    Saves a list of dictionaries to a .jsonl file.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def remove_comments(data):
    """
    Removes the 'Comments' key from each record in the dataset.
    """
    for record in data:
        if 'Comments' in record:
            del record['Comments']
    return data

def get_multilabel_representation(data):
    """
    Creates a multi-label binary representation for each record and returns
    the representation along with the label-to-index mapping.
    """
    all_entity_labels = sorted(list(set(entity['label'] for record in data for entity in record.get('entities', []))))
    all_relation_labels = sorted(list(set(relation['type'] for record in data for relation in record.get('relations', []))))
    
    combined_labels = all_entity_labels + all_relation_labels
    label_to_index = {label: i for i, label in enumerate(combined_labels)}
    num_labels = len(label_to_index)
    
    y = np.zeros((len(data), num_labels), dtype=int)
    
    for i, record in enumerate(data):
        present_labels = set()
        for entity in record.get('entities', []):
            present_labels.add(entity['label'])
        for relation in record.get('relations', []):
            present_labels.add(relation['type'])
            
        for label in present_labels:
            if label in label_to_index:
                y[i, label_to_index[label]] = 1
                
    # Return both the numpy array and the label map
    return y, label_to_index

def log_distribution_report(y_data, label_map, dataset_name):
    """
    Calculates and logs the distribution of labels in a dataset.
    """
    print(f"\n--- Label Distribution Report for: {dataset_name} ---")
    
    if not label_map:
        print("  No labels found to report.")
        return

    # Invert the map for easy lookup from index to label string
    index_to_label = {i: label for label, i in label_map.items()}
    
    # Calculate counts and percentages
    counts = np.sum(y_data, axis=0)
    total_records = len(y_data)
    percentages = (counts / total_records) * 100 if total_records > 0 else [0] * len(counts)

    # Prepare data for printing
    print(f"  Total Records: {total_records}")
    print("  " + "="*50)
    print(f"  {'Label':<25} | {'Count':>10} | {'Percentage':>12}")
    print("  " + "-"*50)
    
    for i in range(len(counts)):
        label_name = index_to_label[i]
        count = counts[i]
        percent = percentages[i]
        print(f"  {label_name:<25} | {count:>10} | {percent:>11.2f}%")
        
    print("  " + "="*50)

def generate_partitions(config):
    """
    Generates stratified training and test partitions based on both entity
    and relation labels, and logs the distribution.
    """
    # --- 1. Load Configuration Parameters ---
    base_dir = Path().cwd()
    input_file = base_dir / config['data']['raw_input_file']
    output_dir_base = base_dir / config['data']['partitions_dir']
    test_set_path = base_dir / config['data']['holdout_test_set_path']
    
    partition_sizes = config['data']['partition_sizes']
    n_samples = config['data']['n_samples']
    base_seed = config['data']['base_seed']
    test_split_ratio = config['data']['test_split_ratio']

    # --- 2. Read and Preprocess the Raw Data ---
    print(f"Reading raw data from: {input_file}")
    raw_data = read_jsonl(input_file)
    cleaned_data = remove_comments(raw_data)
    print(f"Successfully loaded and cleaned {len(cleaned_data)} records.")

    # --- 3. Create Global Train/Test Split using Stratification ---
    print(f"\nCreating a stratified train/test split with a {test_split_ratio:.0%} test size.")
    
    y_all, label_map = get_multilabel_representation(cleaned_data)
    X_indices = np.arange(len(cleaned_data)).reshape(-1, 1)

    stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[1.0 - test_split_ratio, test_split_ratio])
    split1_indices, split2_indices = next(stratifier.split(X_indices, y_all))

    if len(split1_indices) > len(split2_indices):
        train_indices, test_indices = split1_indices, split2_indices
    else:
        train_indices, test_indices = split2_indices, split1_indices

    global_train_data = [cleaned_data[i] for i in train_indices]
    global_test_data = [cleaned_data[i] for i in test_indices]

    save_jsonl(global_test_data, test_set_path)
    print(f"Global test set created with {len(global_test_data)} records. Saved to: {test_set_path}")
    print(f"Global training set created with {len(global_train_data)} records.")

    # --- 4. Log Label Distributions ---
    y_train_global, train_label_map = get_multilabel_representation(global_train_data)
    y_test_global, test_label_map = get_multilabel_representation(global_test_data)
    
    # Use the label map from the full training set as the canonical map
    log_distribution_report(y_train_global, train_label_map, "Global Training Set")
    log_distribution_report(y_test_global, test_label_map, "Holdout Test Set")


    # --- 5. Generate Stratified Training Partition Samples ---
    X_train_indices = np.arange(len(global_train_data)).reshape(-1, 1)

    for size in partition_sizes:
        if size == "all":
            print("\nGenerating partition for size: 'all'")
            output_dir_sample = output_dir_base / "train-all" / "sample-1"
            save_jsonl(global_train_data, output_dir_sample / "train.jsonl")
            print(f"  - Sample 1: Train={len(global_train_data)}. Saved to: {output_dir_sample}")
            continue

        print(f"\nGenerating {n_samples} stratified samples for partition size: {size}")
        
        if size >= len(global_train_data):
            print(f"  - Warning: Requested partition size {size} is >= the global training set size {len(global_train_data)}. "
                  "Creating one sample with all training data instead.")
            output_dir_sample = output_dir_base / f"train-{size}" / "sample-1"
            save_jsonl(global_train_data, output_dir_sample / "train.jsonl")
            print(f"  - Sample 1: Train={len(global_train_data)}. Saved to: {output_dir_sample}")
            continue

        for i in range(n_samples):
            sample_num = i + 1
            random.seed(base_seed + i)
            shuffled_train_indices = list(range(len(global_train_data)))
            random.shuffle(shuffled_train_indices)

            shuffled_X_train = X_train_indices[shuffled_train_indices]
            shuffled_y_train = y_train_global[shuffled_train_indices]

            sample_ratio = size / len(global_train_data)
            
            sample_stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[sample_ratio, 1.0 - sample_ratio])
            
            try:
                sample_split1, sample_split2 = next(sample_stratifier.split(shuffled_X_train, shuffled_y_train))
                
                if abs(len(sample_split1) - size) < abs(len(sample_split2) - size):
                    sample_idx_local = sample_split1
                else:
                    sample_idx_local = sample_split2

                original_indices = [shuffled_train_indices[k] for k in sample_idx_local]
                train_data = [global_train_data[j] for j in original_indices]

            except ValueError as e:
                print(f"  - Error during stratification for sample {sample_num} (size {size}): {e}. Skipping.")
                continue

            output_dir_sample = output_dir_base / f"train-{size}" / f"sample-{sample_num}"
            save_jsonl(train_data, output_dir_sample / "train.jsonl")
            print(f"  - Sample {sample_num} (seed {base_seed + i}): Train={len(train_data)}. Saved to: {output_dir_sample}")

    print("\nData partitioning and sampling complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate stratified data partitions based on a YAML configuration file.")
    parser.add_argument('--config-path', type=str, required=True, help='Path to the YAML configuration file.')
    
    args = parser.parse_args()

    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config_path}'")
        exit(1)
    
    generate_partitions(config)
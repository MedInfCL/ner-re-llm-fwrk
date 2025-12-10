import argparse
import json
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import classification_report as sklearn_classification_report
import numpy as np
import sys
import yaml

def convert_numpy_types(obj):
    """
    Recursively converts numpy number types in a dictionary to native Python types
    to ensure JSON serialization compatibility.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.intc, np.intp, np.int8,
                    np.int16, np.int32, np.int64, np.uint8,
                    np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def load_predictions(file_path: str) -> list:
    """Loads prediction records from a .jsonl file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def _calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two 1D boxes (entities).
    Each box is represented by a tuple (start_offset, end_offset).
    """
    start1, end1 = box1
    start2, end2 = box2

    # Calculate the intersection coordinates
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    # Calculate the length of the intersection
    intersection_length = max(0, intersection_end - intersection_start)

    # Calculate the length of the union
    union_length = (end1 - start1) + (end2 - start2) - intersection_length

    # Avoid division by zero
    if union_length == 0:
        return 0.0

    return intersection_length / union_length


def calculate_ner_metrics(predictions: list, valid_labels: set = None) -> dict:
    """
    Calculates NER metrics using a strict matching strategy.
    """
    entity_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for record in predictions:
        # --- Start of filtering logic ---
        true_entities = [e for e in record.get('true_entities', []) if valid_labels is None or e.get('label') in valid_labels]
        predicted_entities = [e for e in record.get('predicted_entities', []) if valid_labels is None or e.get('label') in valid_labels]
        # --- End of filtering logic ---

        true_entities_set = {(e['start_offset'], e['end_offset'], e['label']) for e in true_entities if isinstance(e, dict) and 'start_offset' in e and 'end_offset' in e and 'label' in e}
        predicted_entities_set = {(e['start_offset'], e['end_offset'], e['label']) for e in predicted_entities if isinstance(e, dict) and 'start_offset' in e and 'end_offset' in e and 'label' in e}
        
        tp = true_entities_set.intersection(predicted_entities_set)
        fp = predicted_entities_set - true_entities_set
        fn = true_entities_set - predicted_entities_set

        for _, _, label in tp:
            entity_metrics[label]['tp'] += 1
        for _, _, label in fp:
            entity_metrics[label]['fp'] += 1
        for _, _, label in fn:
            entity_metrics[label]['fn'] += 1

    # --- The rest of the function remains the same for calculating the report ---
    report = {}
    all_tp, all_fp, all_fn = 0, 0, 0
    total_support = 0

    all_labels_in_data = set(entity_metrics.keys())
    if predictions and 'true_entities' in predictions[0]:
        all_labels_in_data.update({e['label'] for p in predictions for e in p.get('true_entities', []) if isinstance(e, dict) and e.get('label')})
    
    # Use valid_labels for the report if provided, otherwise use all labels found
    sorted_labels = sorted(list(valid_labels if valid_labels is not None else all_labels_in_data))

    for label in sorted_labels:
        tp = entity_metrics[label]['tp']
        fp = entity_metrics[label]['fp']
        fn = entity_metrics[label]['fn']
        support = tp + fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        report[label] = {'precision': precision, 'recall': recall, 'f1-score': f1, 'support': support}
        all_tp += tp; all_fp += fp; all_fn += fn; total_support += support

    if total_support == 0: return report # Return early if no entities were evaluated

    micro_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    micro_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    report['micro avg'] = {'precision': micro_precision, 'recall': micro_recall, 'f1-score': micro_f1, 'support': total_support}

    weighted_precision = sum(report[label]['precision'] * report[label]['support'] for label in sorted_labels) / total_support
    weighted_recall = sum(report[label]['recall'] * report[label]['support'] for label in sorted_labels) / total_support
    weighted_f1 = sum(report[label]['f1-score'] * report[label]['support'] for label in sorted_labels) / total_support
    report['weighted avg'] = {'precision': weighted_precision, 'recall': weighted_recall, 'f1-score': weighted_f1, 'support': total_support}

    return report


def calculate_ner_metrics_relaxed(predictions: list, label_groups: dict, overlap_threshold: float, valid_labels: set = None) -> dict:
    """
    Calculates NER metrics using a relaxed matching strategy, allowing for
    label grouping and partial overlap (IoU), while respecting a list of valid labels.
    """
    entity_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    label_to_group = {label: group_name for group_name, labels in label_groups.items() for label in labels}

    # --- Start of new logic: Determine the full set of labels to process ---
    if valid_labels is not None:
        # The set of labels we care about includes the valid labels from the config
        # AND any labels that are part of a group we are evaluating.
        labels_in_groups = set(label for labels in label_groups.values() for label in labels)
        labels_to_process = valid_labels.union(labels_in_groups)
    else:
        labels_to_process = None
    # --- End of new logic ---

    for record_idx, record in enumerate(predictions):
        # --- Start of modified filtering logic ---
        true_entities = [
            e for e in record.get('true_entities', []) 
            if labels_to_process is None or e.get('label') in labels_to_process
        ]
        pred_entities = [
            e for e in record.get('predicted_entities', []) 
            if labels_to_process is None or e.get('label') in labels_to_process
        ]
        # --- End of modified filtering logic ---

        matched_preds = [False] * len(pred_entities)

        for true_entity in true_entities:
            true_label = true_entity.get('label')
            true_start, true_end = true_entity.get('start_offset'), true_entity.get('end_offset')
            if not all(isinstance(v, int) for v in [true_start, true_end]):
                continue

            true_label_group = label_to_group.get(true_label, true_label)
            best_match_idx, best_iou = -1, -1

            for i, pred_entity in enumerate(pred_entities):
                if matched_preds[i]: continue
                pred_label = pred_entity.get('label')
                pred_start, pred_end = pred_entity.get('start_offset'), pred_entity.get('end_offset')
                if not all(isinstance(v, int) for v in [pred_start, pred_end]):
                    continue
                
                pred_label_group = label_to_group.get(pred_label, pred_label)
                if true_label_group == pred_label_group:
                    iou = _calculate_iou((true_start, true_end), (pred_start, pred_end))
                    if iou >= overlap_threshold and iou > best_iou:
                        best_iou, best_match_idx = iou, i
            
            if best_match_idx != -1:
                entity_metrics[true_label_group]['tp'] += 1
                matched_preds[best_match_idx] = True
            else:
                entity_metrics[true_label_group]['fn'] += 1

        for i, pred_entity in enumerate(pred_entities):
            if not matched_preds[i]:
                pred_label = pred_entity.get('label')
                pred_label_group = label_to_group.get(pred_label, pred_label)
                entity_metrics[pred_label_group]['fp'] += 1

    # --- The rest of the function remains the same for calculating the report ---
    report = {}
    all_tp, all_fp, all_fn, total_support = 0, 0, 0, 0
    
    if valid_labels is not None:
        # If a filter is provided, the report should only contain those labels.
        report_labels = valid_labels
    else:
        # Otherwise, report on all labels and groups for which metrics were calculated.
        report_labels = set(entity_metrics.keys())
    
    sorted_labels = sorted(list(report_labels))

    for label in sorted_labels:
        tp, fp, fn = entity_metrics[label]['tp'], entity_metrics[label]['fp'], entity_metrics[label]['fn']
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        report[label] = {'precision': precision, 'recall': recall, 'f1-score': f1, 'support': support}
        all_tp += tp; all_fp += fp; all_fn += fn; total_support += support

    if total_support == 0: return report

    micro_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    micro_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    report['micro avg'] = {'precision': micro_precision, 'recall': micro_recall, 'f1-score': micro_f1, 'support': total_support}
    
    weighted_precision = sum(report[label]['precision'] * report[label]['support'] for label in sorted_labels) / total_support
    weighted_recall = sum(report[label]['recall'] * report[label]['support'] for label in sorted_labels) / total_support
    weighted_f1 = sum(report[label]['f1-score'] * report[label]['support'] for label in sorted_labels) / total_support
    report['weighted avg'] = {'precision': weighted_precision, 'recall': weighted_recall, 'f1-score': weighted_f1, 'support': total_support}

    return report


def calculate_re_metrics(predictions: list, valid_labels: set = None) -> dict:
    """
    Calculates RE metrics by comparing sets of relation dictionaries.
    This function serves as the unified metric calculator for both fine-tuned and RAG models.

    Args:
        predictions (list): The list of prediction records. Each record must contain
                            'true_relations' and 'predicted_relations' keys.

    Returns:
        dict: A classification report dictionary.
    """
    relation_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for record in predictions:
        # --- Start of new filtering logic ---
        true_relations = [r for r in record.get('true_relations', []) if valid_labels is None or r.get('type') in valid_labels]
        predicted_relations = [r for r in record.get('predicted_relations', []) if valid_labels is None or r.get('type') in valid_labels]
        # --- End of new filtering logic ---

        true_relations_set = {(r['from_id'], r['to_id'], r['type']) for r in true_relations if isinstance(r, dict) and 'from_id' in r and 'to_id' in r and r.get('type')}
        predicted_relations_set = {(r['from_id'], r['to_id'], r['type']) for r in predicted_relations if isinstance(r, dict) and 'from_id' in r and 'to_id' in r and r.get('type')}

        tp = true_relations_set.intersection(predicted_relations_set)
        fp = predicted_relations_set - true_relations_set
        fn = true_relations_set - predicted_relations_set

        for _, _, rel_type in tp:
            relation_metrics[rel_type]['tp'] += 1
        for _, _, rel_type in fp:
            relation_metrics[rel_type]['fp'] += 1
        for _, _, rel_type in fn:
            relation_metrics[rel_type]['fn'] += 1

    # --- Calculate the final report from the aggregated statistics ---
    report = {}
    all_tp, all_fp, all_fn = 0, 0, 0
    total_support = 0

    all_labels_in_data = set(relation_metrics.keys())
    if predictions and 'true_relations' in predictions[0]:
        all_labels_in_data.update({r['type'] for p in predictions for r in p.get('true_relations', []) if isinstance(r, dict) and r.get('type')})
    
    # Use valid_labels for the report if provided, otherwise use all labels found
    sorted_labels = sorted(list(valid_labels if valid_labels is not None else all_labels_in_data))

    for label in sorted_labels:
        tp = relation_metrics[label]['tp']
        fp = relation_metrics[label]['fp']
        fn = relation_metrics[label]['fn']
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        report[label] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }
        all_tp += tp
        all_fp += fp
        all_fn += fn
        total_support += support

    # Calculate micro average
    micro_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    micro_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    report['micro avg'] = {
        'precision': micro_precision,
        'recall': micro_recall,
        'f1-score': micro_f1,
        'support': total_support
    }

    # Calculate weighted average
    if total_support > 0:
        weighted_precision = sum(report[label]['precision'] * report[label]['support'] for label in sorted_labels) / total_support
        weighted_recall = sum(report[label]['recall'] * report[label]['support'] for label in sorted_labels) / total_support
        weighted_f1 = sum(report[label]['f1-score'] * report[label]['support'] for label in sorted_labels) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0
        
    report['weighted avg'] = {
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1-score': weighted_f1,
        'support': total_support
    }

    return report

def aggregate_metrics(reports: list) -> dict:
    """
    Aggregates metrics from multiple reports to calculate mean and standard deviation.

    Args:
        reports (list): A list of classification report dictionaries.

    Returns:
        dict: A dictionary containing the mean and std dev for key metrics.
    """
    if not reports:
        return {}

    # We focus on the 'weighted avg' as the primary summary statistic
    key_metrics = ['precision', 'recall', 'f1-score']

    # Extract the weighted average scores from each report
    weighted_avg_scores = {metric: [] for metric in key_metrics}
    for report in reports:
        if 'weighted avg' in report:
            for metric in key_metrics:
                weighted_avg_scores[metric].append(report['weighted avg'][metric])

    # Calculate mean and standard deviation for each metric
    summary = {}
    for metric in key_metrics:
        scores = weighted_avg_scores[metric]
        if scores:
            summary[metric] = {
                "mean": np.mean(scores),
                "std": np.std(scores)
            }

    return summary

def process_single_file(
    prediction_path: str, 
    eval_type: str, 
    relaxed_eval: bool, 
    label_groups: dict, 
    overlap_threshold: float
) -> dict:
    """Processes a single prediction file and returns its metrics report."""
    predictions = load_predictions(prediction_path)

    if eval_type == 'ner':
        if relaxed_eval:
            print("--- Running in Relaxed NER Evaluation Mode ---")
            report = calculate_ner_metrics_relaxed(predictions, label_groups, overlap_threshold)
        else:
            print("--- Running in Strict NER Evaluation Mode ---")
            report = calculate_ner_metrics(predictions)
    elif eval_type == 're':
        report = calculate_re_metrics(predictions)
    else:
        raise ValueError(f"Unknown evaluation type: '{eval_type}'. Must be 'ner' or 're'.")

    return convert_numpy_types(report)

def main(
    prediction_path: str,
    prediction_dir: str,
    eval_type: str,
    output_path: str,
    config_path: str = None,
    relaxed_eval: bool = False,
    label_groups_str: str = '{}',
    overlap_threshold: float = 0.5
):
    """
    Main function to calculate and save metrics from a prediction file or directory.
    """
    valid_labels = None
    if config_path:
        print(f"--- Loading valid labels from config: {config_path} ---")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Extract the list of entity/relation names from the config structure
            if eval_type == 'ner':
                valid_labels = {entity['name'] for entity in config.get('rag_prompt', {}).get('entity_labels', [])}
            elif eval_type == 're':
                valid_labels = {rel['name'] for rel in config.get('rag_prompt', {}).get('relation_labels', [])}
            if not valid_labels:
                print("Warning: Could not find 'entity_labels' in the specified config. No label filtering will be applied.")
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}. No label filtering will be applied.")
        except Exception as e:
            print(f"Warning: Error loading or parsing config file: {e}. No label filtering will be applied.")

    if relaxed_eval and eval_type != 'ner':
        print("Warning: Relaxed evaluation is only applicable to NER (--type ner). Ignoring relaxed flags.")
        relaxed_eval = False

    try:
        label_groups = json.loads(label_groups_str)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON string provided for --label-groups: {label_groups_str}")
        sys.exit(1)

    process_args = {
        "eval_type": eval_type,
        "valid_labels": valid_labels,
        "relaxed_eval": relaxed_eval,
        "label_groups": label_groups,
        "overlap_threshold": overlap_threshold
    }

    if prediction_dir:
        print(f"--- Calculating Aggregate Metrics for Directory: {prediction_dir} ---")
        prediction_files = list(Path(prediction_dir).glob('*.jsonl'))
        if not prediction_files:
            raise FileNotFoundError(f"No .jsonl prediction files found in '{prediction_dir}'.")

        print(f"Found {len(prediction_files)} prediction files to process.")

        individual_reports = []
        for file_path in prediction_files:
            report = process_single_file(str(file_path), **process_args)
            individual_reports.append({
                "source_file": file_path.name,
                "report": report
            })

        aggregate_summary = aggregate_metrics([r['report'] for r in individual_reports])
        final_report = {"aggregate_summary": aggregate_summary, "individual_reports": individual_reports}
    else:
        print(f"--- Calculating Metrics for File: {prediction_path} ---")
        final_report = process_single_file(prediction_path, **process_args)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_report, f, indent=4)

    print("\n--- Final Metrics Report ---")
    print(json.dumps(final_report, indent=4))
    print(f"\nReport saved successfully to: {output_path}")

# This function now accepts valid_labels
def process_single_file(
    prediction_path: str,
    eval_type: str,
    valid_labels: set,
    relaxed_eval: bool,
    label_groups: dict,
    overlap_threshold: float
) -> dict:
    """Processes a single prediction file and returns its metrics report."""
    predictions = load_predictions(prediction_path)

    if eval_type == 'ner':
        if relaxed_eval:
            print("--- Running in Relaxed NER Evaluation Mode ---")
            report = calculate_ner_metrics_relaxed(predictions, label_groups, overlap_threshold, valid_labels)
        else:
            print("--- Running in Strict NER Evaluation Mode ---")
            report = calculate_ner_metrics(predictions, valid_labels)
    elif eval_type == 're':
        report = calculate_re_metrics(predictions)
    else:
        raise ValueError(f"Unknown evaluation type: '{eval_type}'. Must be 'ner' or 're'.")

    return convert_numpy_types(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate final metrics from a single prediction file or a directory of files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prediction-path', type=str, help='Path to a single .jsonl file containing predictions.')
    group.add_argument('--prediction-dir', type=str, help='Path to a directory containing multiple .jsonl prediction files.')

    parser.add_argument('--type', type=str, required=True, choices=['ner', 're'], help="The type of task evaluation.")
    parser.add_argument('--output-path', type=str, help="Path to save the final JSON metrics report.")
    parser.add_argument('--config-path', type=str, help="Path to the RAG config file to get the list of valid entity labels for filtering.")
    parser.add_argument('--relaxed-eval', action='store_true', help="If set, enables relaxed evaluation for NER.")
    parser.add_argument('--label-groups', type=str, default='{}', help='JSON string for label groups in relaxed evaluation.')
    parser.add_argument('--overlap-threshold', type=float, default=0.5, help='Minimum IoU overlap threshold for relaxed NER evaluation.')
    
    args = parser.parse_args()

    if not args.output_path:
        suffix = "_relaxed" if args.relaxed_eval else ""
        if args.prediction_dir:
            args.output_path = Path(args.prediction_dir) / f'aggregate_metrics{suffix}.json'
        else:
            input_path = Path(args.prediction_path)
            base_name = f"final_metrics_{input_path.stem}"
            args.output_path = input_path.parent / f"{base_name}{suffix}.json"

    try:
        main(
            prediction_path=args.prediction_path,
            prediction_dir=args.prediction_dir,
            eval_type=args.type,
            output_path=args.output_path,
            config_path=args.config_path,
            relaxed_eval=args.relaxed_eval,
            label_groups_str=args.label_groups,
            overlap_threshold=args.overlap_threshold
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
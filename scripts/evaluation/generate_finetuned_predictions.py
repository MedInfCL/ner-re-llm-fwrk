import argparse
import yaml
from pathlib import Path
import json
import torch
import shutil
from datetime import datetime
import numpy as np
import itertools

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader.ner_datamodule import NERDataModule
from src.data_loader.re_datamodule import REDataModule
from src.models.ner_bert import BertNerModel
from src.models.re_model import REModel
from src.evaluation.predictor import Predictor

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

def decode_entities_from_tokens(
    source_text: str,
    token_label_ids: list,
    inv_label_map: dict,
    tokenizer
) -> list:
    """
    Decodes a sequence of token-level integer labels into a list of entity
    dictionaries using the BIO scheme.

    Args:
        source_text (str): The original, untokenized text.
        token_label_ids (list): The list of predicted integer labels for each token.
        inv_label_map (dict): A dictionary mapping integer IDs back to BIO labels.
        tokenizer: The Hugging Face tokenizer instance.

    Returns:
        list: A list of decoded entity dictionaries, each with 'text' and 'label'.
    """
    if not source_text or len(token_label_ids) == 0:
        return []

    # Tokenize to get the mapping from tokens back to character offsets
    tokenization = tokenizer(
        source_text,
        return_offsets_mapping=True,
        max_length=512,
        truncation=True
    )
    offset_mapping = tokenization['offset_mapping']

    reconstructed_entities = []
    current_entity_offsets = []
    current_entity_label = None

    # Align labels with the actual tokens, skipping special tokens
    active_labels = token_label_ids[:len(offset_mapping)]

    for i, label_id in enumerate(active_labels):
        label_str = inv_label_map.get(label_id, "O")
        start_char, end_char = offset_mapping[i]

        # Ignore special tokens like [CLS] and [PAD]
        if end_char == 0:
            continue

        if label_str.startswith("B-"):
            # If an entity was being built, finalize and save it
            if current_entity_label:
                start = current_entity_offsets[0][0]
                end = current_entity_offsets[-1][1]
                entity_text = source_text[start:end]
                reconstructed_entities.append({"text": entity_text, "label": current_entity_label, "start_offset": start, "end_offset": end})
            
            # Start a new entity
            current_entity_offsets = [(start_char, end_char)]
            current_entity_label = label_str.split("-")[1]

        elif label_str.startswith("I-") and current_entity_label == label_str.split("-")[1]:
            # Continue the current entity
            current_entity_offsets.append((start_char, end_char))

        else: # O-tag or a B-tag for a different entity
            if current_entity_label:
                start = current_entity_offsets[0][0]
                end = current_entity_offsets[-1][1]
                entity_text = source_text[start:end]
                reconstructed_entities.append({"text": entity_text, "label": current_entity_label, "start_offset": start, "end_offset": end})
            current_entity_offsets = []
            current_entity_label = None
    
    # Add any lingering entity after the loop finishes
    if current_entity_label:
        start = current_entity_offsets[0][0]
        end = current_entity_offsets[-1][1]
        entity_text = source_text[start:end]
        reconstructed_entities.append({"text": entity_text, "label": current_entity_label, "start_offset": start, "end_offset": end})
        
    return reconstructed_entities


def decode_relations_from_ids(
    record: dict,
    predictions: list,
    inv_relation_map: dict,
    datamodule_relation_map: dict
) -> list:
    """
    Decodes a flat list of integer relation predictions back into a structured
    list of relation dictionaries.

    Args:
        record (dict): The original source data record, containing the 'entities'.
        predictions (list): The flat list of predicted integer labels for this record.
        inv_relation_map (dict): Maps relation IDs back to their string names.
        datamodule_relation_map (dict): The original relation map from the datamodule,
                                        used to filter for valid relation types.

    Returns:
        list: A list of decoded relation dictionaries.
    """
    reconstructed_relations = []
    
    # Re-generate the entity pairs in the same order as the REDataModule to
    # correctly map the flat prediction list back to its corresponding pair.
    entities = record.get("entities", [])
    relations = record.get("relations", [])
    relation_lookup = {(rel["from_id"], rel["to_id"]): rel["type"] for rel in relations}
    
    prediction_idx = 0
    for head, tail in itertools.permutations(entities, 2):
        # We only consider pairs that were valid during training
        relation_type = relation_lookup.get((head["id"], tail["id"]), "No_Relation")
        if relation_type not in datamodule_relation_map:
            continue

        # Check if we have a prediction for this valid pair
        if prediction_idx < len(predictions):
            predicted_label_id = predictions[prediction_idx]
            predicted_label_str = inv_relation_map.get(predicted_label_id, "No_Relation")

            # We only save the relation if the model predicted something other than "No_Relation"
            if predicted_label_str != "No_Relation":
                reconstructed_relations.append({
                    "from_id": head["id"],
                    "to_id": tail["id"],
                    "type": predicted_label_str
                })
            
            prediction_idx += 1
            
    return reconstructed_relations

def run_prediction_and_save(config):
    """
    Main function to run predictions on a test set and save the decoded outputs.
    """
    task = config.get('task')
    if task not in ['ner', 're']:
        raise ValueError("Configuration file must specify a 'task': 'ner' or 're'.")
    

    model_path = config['model_path']
    test_file = config['test_file']
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Initialize Task-Specific Modules ---
    if task == 'ner':
        print("Initializing NER components for prediction...")
        datamodule = NERDataModule(config=config, test_file=test_file)
        inv_label_map = {v: k for k, v in datamodule.label_map.items()}
        # Create a set of valid entity labels from the configuration for filtering
        valid_entity_labels = set(config.get('model', {}).get('entity_labels', []))
        model = BertNerModel(base_model=model_path)
    else: # task == 're'
        print("Initializing RE components for prediction...")
        datamodule = REDataModule(config=config, test_file=test_file)
        model = REModel(base_model=model_path, tokenizer=datamodule.tokenizer)
    
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    # --- Initialize Model and Predictor ---
    print(f"Loading model from: {model_path}")
    predictor = Predictor(model=model, device=device)

    # --- Get Raw Predictions ---
    predictions, true_labels, _ = predictor.predict(test_loader, task_type=task)

    # --- Load Source Text and Format Output ---
    with open(test_file, 'r', encoding='utf-8') as f:
        source_records = [json.loads(line) for line in f]

    output_data = []
    if task == 'ner':
        # NER task maintains its one-to-one logic
        for i, record in enumerate(source_records):
            true_entities_decoded = []
            for entity in record.get("entities", []):
                if entity.get("label") in valid_entity_labels:
                    true_entities_decoded.append({
                        "text": record["text"][entity["start_offset"]:entity["end_offset"]],
                        "label": entity["label"],
                        "start_offset": entity["start_offset"], 
                        "end_offset": entity["end_offset"]   
                    })

            predicted_entities_decoded = decode_entities_from_tokens(
                source_text=record.get("text", ""),
                token_label_ids=predictions[i],
                inv_label_map=inv_label_map,
                tokenizer=datamodule.tokenizer
            )

            output_data.append(convert_numpy_types({
                "source_text": record.get("text", ""),
                "true_entities": true_entities_decoded,
                "predicted_entities": predicted_entities_decoded
            }))
    else: # RE task now correctly groups instances by source text
        instance_idx = 0
        inv_relation_map = {v: k for k, v in datamodule.relation_map.items()}
        for record in source_records:
            # To correctly map the flat list of predictions back to the source record,
            # we must recalculate how many valid instances this record generates.
            num_instances_for_record = 0
            relation_map = datamodule.relation_map
            relations = record.get("relations", [])
            relation_lookup = {(rel["from_id"], rel["to_id"]): rel["type"] for rel in relations}

            for head, tail in itertools.permutations(record.get("entities", []), 2):
                relation_type = relation_lookup.get((head["id"], tail["id"]), "No_Relation")
                if relation_type in relation_map:
                    num_instances_for_record += 1

            # Slice the predictions and labels for the current source record
            record_predictions = predictions[instance_idx : instance_idx + num_instances_for_record]
            record_true_labels = true_labels[instance_idx : instance_idx + num_instances_for_record]

            # Decode both the true and predicted labels into the dictionary format
            predicted_relations_decoded = decode_relations_from_ids(
                record=record,
                predictions=record_predictions,
                inv_relation_map=inv_relation_map,
                datamodule_relation_map=datamodule.relation_map
            )
            
            true_relations_decoded = decode_relations_from_ids(
                record=record,
                predictions=record_true_labels,
                inv_relation_map=inv_relation_map,
                datamodule_relation_map=datamodule.relation_map
            )

            output_data.append(convert_numpy_types({
                "source_text": record.get("text", ""),
                "true_relations": true_relations_decoded,
                "predicted_relations": predicted_relations_decoded
            }))
            instance_idx += num_instances_for_record


    # --- Save Decoded Outputs ---
    filename_prefix = "predictions" # Unified for both tasks
    output_filename = output_dir / f"{filename_prefix}_{Path(model_path).name}.jsonl"

    with open(output_filename, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nPredictions saved to: {output_filename}")
    print(f"Prediction generation for '{Path(model_path).name}' finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a batch of predictions for all models in a given directory and save raw outputs."
    )
    
    parser.add_argument(
        '--config-path', 
        type=str, 
        required=True, 
        help='Path to the YAML configuration file for evaluation.'
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help="Path to the directory containing the trained model samples (e.g., 'output/models/ner/train-50/20240828_150000')."
    )
    
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    model_dir = Path(args.model_dir)
    sample_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('sample-')])

    if not sample_dirs:
        raise FileNotFoundError(f"No 'sample-*' directories found in '{model_dir}'.")

    # Create an output directory that matches the training run's directory name
    run_folder_name = model_dir.name
    output_dir_for_run = Path(base_config['output_dir']) / run_folder_name
    output_dir_for_run.mkdir(parents=True, exist_ok=True)
    
    print(f"All prediction outputs for this run will be saved in: {output_dir_for_run}")
    shutil.copy(args.config_path, output_dir_for_run / "evaluation_config.yaml")

    print(f"Found {len(sample_dirs)} model samples to generate predictions for.")

    for i, sample_path in enumerate(sample_dirs):
        print(f"\n{'='*20} Generating predictions for: {sample_path.name} ({i+1}/{len(sample_dirs)}) {'='*20}")
        
        sample_config = base_config.copy()
        sample_config['model_path'] = str(sample_path)
        sample_config['output_dir'] = str(output_dir_for_run)
        
        run_prediction_and_save(sample_config)

    print("\nBatch prediction generation finished successfully.")
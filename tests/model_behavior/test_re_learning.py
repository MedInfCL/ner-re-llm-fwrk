import pytest
import yaml
import json
import torch
from pathlib import Path
import sys

# Add the project root to the Python path to allow for absolute imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader.re_datamodule import REDataModule
from src.models.re_model import REModel
from src.training.trainer import Trainer
from src.evaluation.predictor import Predictor
from scripts.evaluation.generate_finetuned_predictions import decode_relations_from_ids
from scripts.evaluation.calculate_final_metrics import calculate_re_metrics
import itertools

def test_re_model_can_overfit_on_small_batch(tmp_path):
    """
    Tests if the RE model can achieve a perfect score on a tiny dataset.
    
    This test verifies that the training pipeline is functioning correctly by
    checking if the model can successfully memorize (overfit) a small, clean
    batch of RE data. A perfect or near-perfect score indicates that the
    model's weights are updating and it is capable of learning relationships.
    """
    # --- 1. Define a "Toy" RE Dataset and Configuration ---
    
    # A small, unambiguous dataset for the model to learn
    toy_dataset = [
        {
            "text": "Nódulo periareolar derecho.",
            "entities": [
                {"id": 1, "label": "HALL", "start_offset": 0, "end_offset": 6},
                {"id": 2, "label": "REG", "start_offset": 7, "end_offset": 26}
            ],
            "relations": [{"from_id": 1, "to_id": 2, "type": "ubicar"}]
        },
        {
            "text": "Microcalcificaciones agrupadas.",
            "entities": [
                {"id": 3, "label": "HALL", "start_offset": 0, "end_offset": 20},
                {"id": 4, "label": "CARACT", "start_offset": 21, "end_offset": 30}
            ],
            "relations": [{"from_id": 3, "to_id": 4, "type": "describir"}]
        },
        {
            "text": "Quiste simple.",
            "entities": [
                {"id": 5, "label": "HALL", "start_offset": 0, "end_offset": 6},
                {"id": 6, "label": "CARACT", "start_offset": 7, "end_offset": 13}
            ],
            "relations": [{"from_id": 5, "to_id": 6, "type": "describir"}]
        }
    ]
    
    # A configuration designed to facilitate overfitting
    overfit_config = {
        'seed': 42,
        'trainer': {
            'n_epochs': 50,
            'batch_size': 3,
            'learning_rate': 1e-3,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'device': "cpu"
        },
        'model': {
            'base_model': 'prajjwal1/bert-tiny',
            'relation_labels': ["ubicar", "describir", "No_Relation"]
        }
    }

    # --- 2. Setup Temporary Files ---
    train_file = tmp_path / "train_re.jsonl"
    with open(train_file, 'w') as f:
        for record in toy_dataset:
            f.write(json.dumps(record) + '\n')

    # --- 3. Train the Model ---
    torch.manual_seed(overfit_config['seed'])
    
    datamodule = REDataModule(config=overfit_config, train_file=train_file)
    datamodule.setup()
    
    model = REModel(
        base_model=overfit_config['model']['base_model'],
        n_labels=len(datamodule.relation_map),
        tokenizer=datamodule.tokenizer
    )
    
    trainer = Trainer(model=model, datamodule=datamodule, config=overfit_config)
    trainer.train()

    # --- 4. Evaluate on the Training Data ---
    print("\n--- Evaluating RE model directly from memory ---")
    predictor = Predictor(model=model, device=torch.device("cpu"))
    
    # The REDataModule creates all permutations, so we use it for evaluation
    datamodule.test_file = train_file
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()
    
    predictions, true_labels, _ = predictor.predict(test_loader, task_type='re')
    
    # --- 5. Assert High Performance ---
    # The raw predictions must be decoded back into the unified dictionary format.
    instance_idx = 0
    all_true_relations = []
    all_predicted_relations = []

    inv_relation_map = {v: k for k, v in datamodule.relation_map.items()}

    for record in toy_dataset:
        # Calculate how many valid instances this record generated
        num_instances_for_record = 0
        relations = record.get("relations", [])
        relation_lookup = {(rel["from_id"], rel["to_id"]): rel["type"] for rel in relations}
        for head, tail in itertools.permutations(record.get("entities", []), 2):
            relation_type = relation_lookup.get((head["id"], tail["id"]), "No_Relation")
            if relation_type in datamodule.relation_map:
                num_instances_for_record += 1

        # Slice the flat prediction lists for the current record
        record_predictions = predictions[instance_idx : instance_idx + num_instances_for_record]
        record_true_labels = true_labels[instance_idx : instance_idx + num_instances_for_record]

        # Decode both true and predicted relations
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

        all_predicted_relations.extend(predicted_relations_decoded)
        all_true_relations.extend(true_relations_decoded)

        instance_idx += num_instances_for_record

    # Now, calculate metrics on the decoded, unified lists
    prediction_records = [{
        "true_relations": all_true_relations,
        "predicted_relations": all_predicted_relations
    }]

    report = calculate_re_metrics(prediction_records)

    print("\n--- Overfitting Sanity Check Metrics (RE) ---")
    print(json.dumps(report, indent=2))

    assert report['weighted avg']['f1-score'] >= 0.95, \
        "Model did not achieve a very high F1-score on the training data, indicating a potential learning issue for RE."
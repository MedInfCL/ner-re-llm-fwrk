# tests/model_behavior/test_ner_learning.py
import pytest
import yaml
import json
import torch
from pathlib import Path
import sys

# Add the project root to the Python path to allow for absolute imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader.ner_datamodule import NERDataModule
from src.models.ner_bert import BertNerModel
from src.training.trainer import Trainer
from src.evaluation.predictor import Predictor
from scripts.evaluation.generate_finetuned_predictions import decode_entities_from_tokens
from scripts.evaluation.calculate_final_metrics import calculate_ner_metrics

def test_ner_model_can_overfit_on_small_batch(tmp_path):
    """
    Tests if the NER model can achieve a perfect score on a tiny dataset.
    
    This test verifies that the training pipeline is functioning correctly by
    checking if the model can successfully memorize (overfit) a small, clean
    batch of data when trained for enough epochs. A perfect or near-perfect
    score indicates that the model's weights are updating and it is capable of learning.
    """
    # --- 1. Define a "Toy" Dataset and Configuration ---
    
    # A small, unambiguous dataset for the model to learn
    toy_dataset = [
        {"text": "The patient has a nodule.", "entities": [{"label": "FIND", "start_offset": 18, "end_offset": 24}]},
        {"text": "We see a mass in the left breast.", "entities": [{"label": "FIND", "start_offset": 9, "end_offset": 13}, {"label": "LOC", "start_offset": 21, "end_offset": 32}]},
        {"text": "Another finding in the upper quadrant.", "entities": [{"label": "FIND", "start_offset": 8, "end_offset": 15}, {"label": "LOC", "start_offset": 23, "end_offset": 37}]},
        {"text": "A cyst was found in the right axilla.", "entities": [{"label": "FIND", "start_offset": 2, "end_offset": 6}, {"label": "LOC", "start_offset": 24, "end_offset": 36}]}
    ]
    
    # A configuration designed to facilitate overfitting on the small dataset
    overfit_config = {
        'seed': 42,
        'trainer': {
            'n_epochs': 50, # More epochs to ensure memorization
            'batch_size': 4,
            'learning_rate': 1e-3, # Higher learning rate for faster convergence
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'device': "cpu"
        },
        'model': {
            'base_model': 'prajjwal1/bert-tiny',
            'entity_labels': ["FIND", "LOC"]
        }
    }

    # --- 2. Setup Temporary Files and Directories ---
    train_file = tmp_path / "train.jsonl"
    with open(train_file, 'w') as f:
        for record in toy_dataset:
            f.write(json.dumps(record) + '\n')
            
    model_output_dir = tmp_path / "model_output"
    model_output_dir.mkdir()

    # --- 3. Train the Model ---
    torch.manual_seed(overfit_config['seed'])
    
    # Initialize and set up the data module
    datamodule = NERDataModule(config=overfit_config, train_file=train_file)
    datamodule.setup()
    
    # Initialize the model
    model = BertNerModel(
        base_model=overfit_config['model']['base_model'],
        n_labels=len(datamodule.label_map)
    )
    
    # Initialize and run the trainer
    trainer = Trainer(model=model, datamodule=datamodule, config=overfit_config)
    trainer.train()
    trainer.save_model(str(model_output_dir))

    # --- 4. Evaluate on the Training Data ---
    
    # Use the model object that was just trained directly, bypassing the save/load step
    print("\n--- Evaluating model directly from memory ---")
    predictor = Predictor(model=model, device=torch.device("cpu"))
        
    # Reuse the same datamodule for evaluation to ensure tokenizer consistency
    datamodule.test_file = train_file
    datamodule.setup(stage='test') # Re-run setup to create the test_dataset
    test_loader = datamodule.test_dataloader()
    
    # Get raw token-level predictions
    predictions, _, _ = predictor.predict(test_loader, task_type='ner')
    
    # --- 5. Assert High Performance ---
    
    inv_label_map = {v: k for k, v in datamodule.label_map.items()}
    
    # Format predictions into the unified entity format
    results = []
    for i, record in enumerate(toy_dataset):
        predicted_entities = decode_entities_from_tokens(
            source_text=record["text"],
            token_label_ids=predictions[i],
            inv_label_map=inv_label_map,
            tokenizer=datamodule.tokenizer
        )
        
        # In this test, true entities are decoded directly from the source for simplicity
        # In this test, true entities are decoded directly from the source for simplicity
        true_entities = [
            {
                "text": record["text"][e["start_offset"]:e["end_offset"]],
                "label": e["label"],
                "start_offset": e["start_offset"],
                "end_offset": e["end_offset"]
            }
            for e in record["entities"]
        ]

        print(f"\nRecord {i+1}:")
        print("True Entities:", true_entities)
        print("Predicted Entities:", predicted_entities)

        results.append({
            "true_entities": true_entities,
            "predicted_entities": predicted_entities
        })

    # Calculate metrics
    report = calculate_ner_metrics(results)
    
    print("\n--- Overfitting Sanity Check Metrics ---")
    print(json.dumps(report, indent=2))
    
    # Assert that the model achieved a perfect score on the data it was trained on
    assert report['weighted avg']['f1-score'] >= 0.95, \
        "Model did not achieve a very high F1-score on the training data, indicating a potential learning issue."
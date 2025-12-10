# scripts/run_re_training.py
import argparse
import yaml
from pathlib import Path
import torch
import sys
from datetime import datetime
import shutil

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader.re_datamodule import REDataModule
from src.models.re_model import REModel
from src.training.trainer import Trainer

def run_batch_re_training(config_path, partition_dir):
    """
    Main function to run the RE training process for all samples in a partition directory.

    Args:
        config_path (str): Path to the RE training YAML configuration file.
        partition_dir (str): Path to the directory containing training samples
                             (e.g., 'data/processed/train-50').
    """
    # --- 1. Load Configuration and Find Samples ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base_partition_dir = Path(partition_dir)
    sample_dirs = sorted([d for d in base_partition_dir.iterdir() if d.is_dir() and d.name.startswith('sample-')])

    if not sample_dirs:
        print(f"Warning: No sample directories found in '{partition_dir}'. Exiting.")
        return

    print(f"Found {len(sample_dirs)} samples to process for RE training in '{base_partition_dir.name}'.")

    # --- 2. Create Timestamped Parent Directory for the Run ---
    base_output_dir = Path(config['paths']['output_dir'])
    data_partition_name = base_partition_dir.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # This is the unique parent directory for this entire batch training run
    run_folder_name = f"{data_partition_name}_{timestamp}"
    run_specific_output_dir = base_output_dir / "re" / run_folder_name
    run_specific_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"All models for this run will be saved in subfolders of: {run_specific_output_dir}")

    # Save a copy of the training configuration for reproducibility
    shutil.copy(config_path, run_specific_output_dir / "training_re_config.yaml")


    # --- 3. Loop Through Each Sample and Train a Model ---
    for sample_dir in sample_dirs:
        train_file_path = sample_dir / "train.jsonl"
        if not train_file_path.exists():
            print(f"  - Skipping {sample_dir.name}: 'train.jsonl' not found.")
            continue

        print(f"\n{'='*20} Starting RE Training for: {sample_dir.name} {'='*20}")

        # Set the seed for reproducibility for each run
        torch.manual_seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed'])

        # --- Initialize Data Module for RE ---
        print(f"Loading RE data from: {train_file_path}")
        datamodule = REDataModule(config=config, train_file=train_file_path)
        datamodule.setup()
        
        n_labels = len(datamodule.relation_map)
        print(f"Number of relation labels in the dataset: {n_labels}")

        # --- Initialize Model for RE ---
        print(f"Initializing RE model: {config['model']['base_model']}")
        model = REModel(
            base_model=config['model']['base_model'],
            n_labels=n_labels,
            tokenizer=datamodule.tokenizer
        )

        # --- Initialize Trainer ---
        print("Initializing trainer...")
        trainer = Trainer(
            model=model,
            datamodule=datamodule,
            config=config
        )

        # --- Start Training ---
        trainer.train()

        # --- Save the final model ---
        sample_name = sample_dir.name
        final_output_dir = run_specific_output_dir / sample_name
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving final RE model to: {final_output_dir}")
        trainer.save_model(final_output_dir)
        
        print(f"\n{'='*20} Finished RE Training for: {sample_dir.name} {'='*20}")

    print("\nAll RE training experiments finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a batch of Relation Extraction training experiments for all samples within a given data partition."
    )
    
    parser.add_argument(
        '--config-path', 
        type=str, 
        required=True, 
        help='Path to the YAML configuration file for RE training.'
    )

    parser.add_argument(
        '--partition-dir',
        type=str,
        required=True,
        help="Path to the directory containing the training samples (e.g., 'data/processed/train-50')."
    )
    
    args = parser.parse_args()
    
    run_batch_re_training(args.config_path, args.partition_dir)
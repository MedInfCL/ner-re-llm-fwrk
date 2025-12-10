# Training Scripts

## Overview

This directory contains the high-level scripts used to launch training experiments for both **Named Entity Recognition (NER)** and **Relation Extraction (RE)** models. These scripts are the primary entry points for the training workflow, designed to be run from the command line.

They are built to handle **batch training**, meaning they can automatically find and process multiple training data samples within a given partition directory (e.g., all 5 samples in `data/processed/train-50/`), training a separate model for each one. This is essential for the project's goal of evaluating model performance across different data subsets.

-----

## `run_ner_training.py`

### Purpose

This script manages the end-to-end training process for NER models. It initializes the necessary components (`NERDataModule`, `BertNerModel`, `Trainer`), orchestrates the training loop, and saves the final model artifacts.

### Workflow

1.  **Configure Training**: Ensure the appropriate YAML is prepared. Use `configs/training_ner_config.yaml` for NER runs and `configs/training_re_config.yaml` for RE runs. These files control hyperparameters (base model, learning rate, batch size, epochs, output paths, etc.).

2.  **Execute the Script**: Run the script from the repository root and pass the config path plus the partition directory that contains `sample-*` subfolders. The scripts require two arguments: `--config-path` (the YAML) and `--partition-dir` (e.g., `data/processed/train-50`).

    ```powershell
    # NER: run training for all samples under the partition dir
    python scripts/training/run_ner_training.py --config-path configs/training_ner_config.yaml --partition-dir data/processed/train-50

    # RE: run training for all samples under the partition dir
    python scripts/training/run_re_training.py --config-path configs/training_re_config.yaml --partition-dir data/processed/train-50
    ```

    Optional: many training configs also support overriding a few values at the command line inside the script wrapper (check the script header); when in doubt, edit the YAML and re-run.

3.  **Output layout**: The script creates a timestamped run folder under the configured `output/models/<task>/<partition>/` path. Inside that run folder, each sample's trained model is saved in `sample-*` directories, e.g.:

    - `output/models/ner/train-50/2025MMDD_HHMMSS/sample-1/`
      - Contains model weights, tokenizer files, and a copy of the training config used for that run.

Notes and recommendations:

- Partition directory format: the scripts expect `sample-*` subdirectories inside the partition directory. Each `sample-*` should include a `train.jsonl` file used for that sample.
- Ensure `model.entity_labels` (NER) or `model.relation_labels` (RE) are present in the config when needed; the datamodules use these to build label maps. For RE training, include a `No_Relation` label if you intend to train negative pairs.
- Device selection: the trainer will pick CUDA if available; verify CUDA/driver availability before launching large runs.
- Checkpoints and reproducibility: each run writes a copy of the YAML into the run folder for traceability.

-----

## `run_re_training.py`

### Purpose

This script manages the training process for RE models. It is structurally similar to the NER script but is tailored for relation extraction, initializing the `REDataModule` and `REModel`.

### Workflow

1.  **Configure Training**: Ensure the `configs/training_re_config.yaml` file is configured for the RE task.

2.  **Execute the Script**: Run the script from the **root directory**, pointing to the RE configuration and the desired data partition.

    ```bash
    python scripts/training/run_re_training.py \
      --config-path configs/training_re_config.yaml \
      --partition-dir data/processed/train-50
    ```

3.  **Output**: The script creates a unique, timestamped directory for the entire run, inside which each trained model sample will be saved. This prevents accidental overwriting of previous results. For example: `output/models/re/train-50/20240828_120729/sample-1/`.
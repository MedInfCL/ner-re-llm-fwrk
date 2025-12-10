# Evaluation Module

## Overview

This directory contains the script responsible for running model inference and generating predictions for both **Named Entity Recognition (NER)** and **Relation Extraction (RE)** tasks. The primary goal of this module is to provide a standardized, task-agnostic interface for evaluating a trained model on a test dataset.

---

## `predictor.py`

### Purpose

The `Predictor` class encapsulates the logic for loading a fine-tuned model, running it in evaluation mode, and processing the model's output logits to produce clean, usable predictions. It is designed to work seamlessly with the `DataLoader` instances provided by the `NERDataModule` or `REDataModule`.

### Key Features

-   **Task-Agnostic Design**: A single `predict` method handles both NER and RE tasks. It dynamically adapts its prediction logic based on a `task_type` parameter, correctly interpreting either token-level (NER) or sequence-level (RE) logits.
-   **Device Management**: Automatically moves the model and data batches to the appropriate device (`cuda` or `cpu`), ensuring efficient inference.
-   **Batch Processing**: Iterates through a `DataLoader`, processes batches of data, and accumulates predictions and true labels.
-   **Label Alignment (NER)**: For NER tasks, it correctly aligns the predicted token labels with the true labels, filtering out any padded tokens (identified by the label `-100`) to ensure that metrics are calculated only on the actual sequence content.
-   **Clean Output**: Returns two simple lists—one for all predictions and one for all corresponding true labels—which can be directly consumed by standard evaluation libraries like `seqeval` or `scikit-learn`.

### Small implementation notes

- Return contract for `Predictor.predict`:
	- `all_predictions`: list — for NER this is a list of numpy arrays (one per batch/record) containing token-level predicted label IDs; for RE this is a flat list/array of predicted label IDs.
	- `all_true_labels`: list — for NER this is a list of lists containing true token label IDs with padded positions (`-100`) removed; for RE this is a flat list/array of true labels.
	- `all_logits`: numpy.ndarray — raw logits stacked across batches (useful for calibration or further analysis).

- Batch keys / label naming: `Predictor.predict` expects NER batches to contain a `labels` tensor and RE batches to contain a `label` tensor (singular). Custom dataloaders should follow this convention to be compatible.

- Empty-dataloader behavior: if the provided `test_dataloader` yields no batches the function returns `([], [], np.array([]))`.

- Device and model state: the `Predictor` constructor calls `model.to(device)` and `model.eval()`. If you pass a shared model object and plan to continue training after prediction, remember to call `model.train()` afterwards.
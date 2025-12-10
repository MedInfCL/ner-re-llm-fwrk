# Model Architectures

## Overview

This directory contains the PyTorch model definitions used for the **Named Entity Recognition (NER)** and **Relation Extraction (RE)** tasks. The classes defined here act as wrappers around standard Hugging Face `transformers` models, simplifying their initialization and use within the project's training and evaluation pipelines.

Each model class is a `torch.nn.Module` that provides a consistent interface for the `Trainer` and `Predictor` modules.

---

## `ner_bert.py`

### Purpose

The `BertNerModel` class is a wrapper for the `AutoModelForTokenClassification` from the Hugging Face library. It is specifically designed for NER tasks, where the goal is to predict a label for each token in a sequence.

### Key Features

-   **Simplified Interface**: Abstracts the details of model configuration and initialization, providing a clean API for the rest of the application.
-   **Dual-Mode Initialization**:
    -   **Training**: When an `n_labels` argument is provided, it initializes a pre-trained base model with a new, randomly initialized token classification head suited for the specified number of entity labels.
    -   **Evaluation**: When `n_labels` is `None`, it loads a complete, fine-tuned model directly from a specified path.
-   **Standard Forward Pass**: The `forward` method directly calls the underlying Hugging Face model, passing `input_ids`, `attention_mask`, and `labels` to compute loss and logits.

---

## `re_model.py`

### Purpose

The `REModel` class is a wrapper for the `AutoModelForSequenceClassification` model, adapted for the task of Relation Extraction. In this project, RE is formulated as a classification problem where the model predicts the relationship type between a pair of entities in a given sequence.

### Key Features

-   **Sequence Classification for RE**: Leverages a sequence classification architecture, which is effective for classifying the relationship based on text containing marked entity pairs.
-   **Embedding Resizing**: During training, it automatically resizes the model's token embeddings to accommodate the special entity marker tokens (e.g., `[E1_START]`, `[E1_END]`) added by the `REDataModule`. This is a critical step for the marker-based RE approach.
-   **Dual-Mode Initialization**:
    -   **Training**: When `n_labels` and a `tokenizer` are provided, it initializes a base model with a new classification head and adjusts embeddings.
    -   **Evaluation**: When `n_labels` is `None`, it loads a fine-tuned RE model directly from a path.

---

## Implementation notes

- `save_pretrained`: Both `BertNerModel` and `REModel` expose a `save_pretrained(output_dir)` method that delegates to the underlying Hugging Face model's `save_pretrained`. This is what the training pipeline uses to persist checkpoints.

- RE tokenizer requirement: When initializing `REModel` for training (i.e., with `n_labels` provided), a valid `tokenizer` must be passed to the constructor so that the model can resize its token embeddings for the special entity marker tokens (the constructor raises a `ValueError` if `tokenizer` is missing).

- Forward outputs: Both model wrappers return the Hugging Face model's output objects (`TokenClassifierOutput` for NER, `SequenceClassifierOutput` for RE). These objects contain `.logits` and, when `labels` are provided, `.loss` as well. The rest of the training/evaluation code expects this standard interface.
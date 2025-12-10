# Training Module

## Overview

This directory contains the core logic for executing the model training loop. It is designed to be reusable and task-agnostic, capable of training any model that adheres to the `torch.nn.Module` interface and is compatible with the project's data modules.

---

## `trainer.py`

### Purpose

The `Trainer` class encapsulates the entire training process for a given model and dataset. It handles all the standard boilerplate associated with modern training loops, such as setting up optimizers, learning rate schedulers, and iterating over data, while providing clear logging and progress tracking.

### Key Features

-   **Standardized Training Loop**: Implements a complete training loop, iterating through epochs and batches, performing forward and backward passes, and updating model weights.
-   **Device Management**: Automatically detects and utilizes an available GPU (`cuda`), falling back to the CPU if one is not present. It ensures that the model and all data batches are moved to the correct device for training.
-   **Optimizer and Scheduler Creation**:
    -   **Optimizer**: Configures the AdamW optimizer, applying weight decay to all parameters except for biases and LayerNorm weights, a standard practice for training transformers.
    -   **Scheduler**: Creates a linear learning rate scheduler with a warmup period, which gradually increases the learning rate at the beginning of training before linearly decaying it. This helps stabilize training.
-   **Gradient Clipping**: Applies gradient clipping (`clip_grad_norm_`) during the backward pass to prevent exploding gradients, further enhancing training stability.
-   **Task-Agnostic Label Handling**: Dynamically determines the correct key for labels in a batch (either `'labels'` for NER or `'label'` for RE), allowing it to work with both data modules seamlessly.
-   **Progress Tracking**: Uses `tqdm` to display a progress bar for each epoch, showing the running loss for immediate feedback on training performance.
-   **Model Persistence**: Includes a `save_model` method that saves the trained model's weights and the associated tokenizer configuration to a specified directory, ensuring the model can be easily reloaded for evaluation.

---

## Implementation notes

- Device selection: the `Trainer` uses the device string from `config['trainer']['device']` if `torch.cuda.is_available()`; otherwise it falls back to `cpu`. The trainer moves the model to `device` at construction time.

- Empty dataloader handling: if the training dataloader has length 0 the trainer prints a warning and skips training.

- Optimizer details: the trainer creates an `AdamW` optimizer and applies `weight_decay` to all parameters except those whose names contain `'bias'` or `'LayerNorm.weight'`. The learning rate is read from `config['trainer']['learning_rate']`.

- Scheduler and warmup: the linear scheduler with warmup is created using `get_linear_schedule_with_warmup`. `total_steps` is computed as `len(train_dataloader) * n_epochs`, and the number of warmup steps is `int(total_steps * warmup_ratio)`. If `total_steps == 0` a no-op LambdaLR is returned.

- Gradient clipping: gradients are clipped with `clip_grad_norm_` to `1.0` before the optimizer step.

- Label key handling: the training loop accepts batches where the labels tensor is named either `labels` (NER) or `label` (RE); it picks the correct key automatically.

- Saving models: `save_model(output_dir)` calls `self.model.save_pretrained(output_dir)` and `self.datamodule.tokenizer.save_pretrained(output_dir)` to persist both model weights and tokenizer artifacts used during training.
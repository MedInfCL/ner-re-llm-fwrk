# Data Loader Modules

## Overview

This directory contains the data loading and preprocessing modules for both **Named Entity Recognition (NER)** and **Relation Extraction (RE)** tasks. These modules are responsible for transforming raw `.jsonl` data into tokenized and formatted tensors suitable for training and evaluation with Hugging Face models.

Each module consists of two primary classes:
1.  A `Dataset` class (`NERDataset`, `REDataset`) that handles the core logic of reading, parsing, and transforming individual data records.
2.  A `DataModule` class (`NERDataModule`, `REDataModule`) that orchestrates the data loading process, including tokenizer initialization, dataset setup, and providing `DataLoader` instances.

---

## `ner_datamodule.py`

### Purpose

This module is designed for **token-level classification** tasks. It reads text and a list of character-level entity annotations and produces tokenized inputs where each token is assigned a label (e.g., `B-FIND`, `I-FIND`, `O`).

### Key Features

-   **BIO Labeling**: Automatically converts character-based entity spans into the standard BIO (Beginning, Inside, Outside) tagging scheme.
-   **Subword Alignment**: Aligns character-based entity spans with their corresponding tokens using the tokenizer's offset mapping. This correctly labels all subwords belonging to an entity and assigns special tokens (like `[CLS]` and `[PAD]`) a label of `-100` to be ignored during loss calculation.
-   **Dynamic Label Mapping**: Constructs a label-to-ID map from the list of entity types provided in the configuration, ensuring flexibility.
-   **Robust Validation**: Includes checks to handle records with missing or malformed entities and warns the user about entity labels present in the data but not specified in the configuration.

### Example Workflow

The main challenge in NER is to convert character-level entity spans into token-level labels, especially when words are split into subword tokens.

1.  **Original Data (`.jsonl` line)**
    The input is a JSON object with the text and the start/end character positions of an entity.

    ```json
    {
      "text": "Review for microcalcifications.",
      "entities": [
        {"label": "FIND", "start_offset": 11, "end_offset": 30}
      ]
    }
    ```

2.  **Tokenization**
    The tokenizer may split a single word into multiple subword tokens. The `offset_mapping` keeps track of the original character span for each token.

    * Original Word: `microcalcifications`
    * Tokens: `['[CLS]', 'review', 'for', 'micro', '##ca', '##lci', '##fic', '##ations', '.', '[SEP]']`

3.  **Label Alignment**
    Using the character offsets, the dataloader assigns the correct BIO label to each token. The first token of the entity gets the `B-` (Beginning) tag, and all subsequent subword tokens of the same entity get the `I-` (Inside) tag.

    | Token | Label |
    | :--- | :--- |
    | `[CLS]` | -100 |
    | `review` | O |
    | `for` | O |
    | `micro` | B-FIND |
    | `##ca` | I-FIND |
    | `##lci` | I-FIND |
    | `##fic` | I-FIND |
    | `##ations` | I-FIND |
    | `.` | O |
    | `[SEP]` | -100 |

This aligned list of labels is then converted to integer IDs and passed to the model for training.

---

## `re_datamodule.py`

### Purpose

This module is tailored for **sequence-level classification** to identify relationships between pairs of entities. It transforms each record into multiple instances, where each instance represents a single, directional pair of entities from the original text.

### Key Features

-   **Entity Pair Generation**: Creates all possible ordered pairs of entities within a single text record (`itertools.permutations`), generating a distinct training instance for each pair.
-   **Special Token Markers**: Inserts special tokens (e.g., `[E1_START]`, `[E1_END]`, `[E2_START]`, `[E2_END]`) around the head and tail entities of a pair. This provides strong, explicit positional signals to the model, helping it focus on the relationship between the two marked entities.
-   **"No\_Relation" Handling**: Assigns a "No\_Relation" label to entity pairs that are not explicitly related in the dataset, allowing the model to learn to distinguish between related and unrelated pairs.
-   **Configuration-Driven Filtering**: Ignores relation types found in the data but not defined in the configuration `relation_labels`, issuing a warning to the user.

### Example Workflow

To understand how the special tokens are used, consider the following record:

1.  **Original Data (`.jsonl` line)**
    The input is a single line of JSON containing the text and entity annotations.

    ```json
    {
      "text": "Nódulo en mama derecha.",
      "entities": [
        {"id": 1, "label": "HALL", "start_offset": 0, "end_offset": 6},
        {"id": 2, "label": "REG", "start_offset": 10, "end_offset": 22}
      ],
      "relations": [
        {"from_id": 1, "to_id": 2, "type": "ubicar"}
      ]
    }
    ```

2.  **Instance Creation**
    The dataloader creates an instance for each potential relationship. Let's consider the pair where "Nódulo" is the head entity (E1) and "mama derecha" is the tail entity (E2).

3.  **Marker Insertion**
    Before tokenization, special markers are inserted into the raw text to bracket the head and tail entities for this specific instance.

    * Original Text: `Nódulo en mama derecha.`
    * Marked Text for Model: `[E1_START]Nódulo[E1_END] en [E2_START]mama derecha[E2_END].`

This new, marked string is then passed to the tokenizer. The model learns to use the `[E1_START]...[E2_END]` markers to understand which pair of entities it should classify for the `ubicar` relationship.

---

## Implementation notes and configuration caveats

- RE "No_Relation" behavior: the `REDataset` defaults to assigning the string `"No_Relation"` to pairs that are not present in the `relations` list of a record. However, the dataset will only include instances whose `relation_type` is present in `config['model']['relation_labels']`. That means you must include `"No_Relation"` in `model.relation_labels` if you want negative (no-relation) examples to be part of the training data; otherwise those pairs will be skipped.

- Required config keys:
  - NER: provide `model.base_model` (or `model_path`) and, preferably, `model.entity_labels` (list of entity names).
  - RE: provide `model.base_model` (or `model_path`) and `model.relation_labels` (list of relation names). The RE datamodule raises an error if no model path is found.

- NER label fallback: `NERDataModule` looks for `config['model']['entity_labels']` to construct the BIO label map. If that key is missing a fallback list (for example `['FIND','REG','OBS','GANGLIOS']`) will be used — it's recommended to explicitly set `model.entity_labels` in your config to match your dataset.

- Batch size configuration:
  - Training batch size is read from `trainer.batch_size` in the config (default 8).
  - Test/evaluation batch size is read from the top-level `batch_size` key in the config (default 16).

- Tokenizers and special tokens:
  - Special tokens are detected using the tokenizer `offset_mapping` (tokens with offsets `(0, 0)` are treated as special) and are assigned label `-100` for NER so they're ignored by the loss function. The concrete special token names (e.g., `[CLS]`, `[SEP]`) depend on the chosen tokenizer.
  - The RE datamodule adds additional special tokens (`[E1_START]`, `[E1_END]`, `[E2_START]`, `[E2_END]`) to the tokenizer via `tokenizer.add_special_tokens` so those markers are recognized during tokenization.
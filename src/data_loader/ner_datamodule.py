import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import warnings

class NERDataset(Dataset):
    """
    A PyTorch Dataset for Named Entity Recognition tasks.
    It handles tokenization and alignment of labels with tokens.
    """

    def __init__(self, file_path, tokenizer, label_map, warned_entities_set=None):
        """
        Args:
            file_path (str): Path to the .jsonl data file.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer for encoding text.
            label_map (dict): A mapping from entity labels (str) to integer IDs.
            warned_entities_set (set, optional): A set to track entities that have already triggered a warning.
        """
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.data = self._load_data(file_path)
        self.warned_entities = warned_entities_set if warned_entities_set is not None else set()

    def _load_data(self, file_path):
        """Loads data from a .jsonl file."""
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        return records

    def __len__(self):
        """Returns the number of records in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single data record.

        Args:
            idx (int): The index of the record to retrieve.

        Returns:
            dict: A dictionary containing 'input_ids', 'attention_mask', and 'labels'.
        """
        record = self.data[idx]
        text = record["text"]
        entities = record.get("entities", [])

        # Tokenize the text and request the offset mapping, which provides
        # the start and end character positions for each token.
        tokenized_inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True # Request character offsets for each token
        )

        input_ids = tokenized_inputs["input_ids"].squeeze()
        attention_mask = tokenized_inputs["attention_mask"].squeeze()
        
        # The offset mapping is used to align character-based entity labels
        # with the token-based input for the model.
        offset_mapping = tokenized_inputs["offset_mapping"].squeeze()

        labels = self._align_labels(offset_mapping, entities)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _align_labels(self, offset_mapping, entities):
        """
        Aligns entity labels with the tokenized input using the offset mapping.

        Args:
            offset_mapping (torch.Tensor): A tensor where each element is a pair
                                           of (start_char, end_char) for a token.
            entities (list): A list of entity dictionaries from the raw data.

        Returns:
            torch.Tensor: A tensor of label IDs aligned with the input tokens.
        """
        # Initialize labels with -100, a value ignored by the loss function,
        # for all positions including special tokens like [CLS] and [SEP].
        labels = torch.full(offset_mapping.shape[:1], fill_value=-100, dtype=torch.long)

        for entity in entities:
            if not isinstance(entity, dict):
                raise TypeError(f"Entity must be a dictionary, but got: {type(entity)}")

            entity_label = entity["label"]
            
            # Check if the entity type is one that the model is configured to recognize.
            if f"B-{entity_label}" not in self.label_map:
                if entity_label not in self.warned_entities:
                    warnings.warn(f"Entity label '{entity_label}' not found in config and will be ignored.")
                    self.warned_entities.add(entity_label)
                continue

            start_char, end_char = entity["start_offset"], entity["end_offset"]

            if not isinstance(start_char, int) or not isinstance(end_char, int):
                raise TypeError(
                    f"Entity offsets must be integers. "
                    f"Got start_offset: {start_char} (type {type(start_char)}) and "
                    f"end_offset: {end_char} (type {type(end_char)})."
                )

            if start_char >= end_char:
                continue

            # Find all tokens whose character spans fall within the entity's span.
            is_first_token = True
            for i, (token_start, token_end) in enumerate(offset_mapping):
                # Skip special tokens, which have an offset of (0, 0).
                if token_start == 0 and token_end == 0:
                    continue

                # A token is part of the entity if its span overlaps with the entity's span.
                # The condition is max(start) < min(end).
                if max(token_start, start_char) < min(token_end, end_char):
                    # Assign the "B-" (Beginning) tag to the first token of an entity.
                    if is_first_token:
                        labels[i] = self.label_map[f"B-{entity_label}"]
                        is_first_token = False
                    # Assign the "I-" (Inside) tag to subsequent tokens of the same entity.
                    else:
                        labels[i] = self.label_map[f"I-{entity_label}"]

        # Only change the label for tokens that are not special characters
        # and were not assigned an entity label.
        for i, (token_start, token_end) in enumerate(offset_mapping):
            # If the token is not a special token ([CLS], [SEP], etc.) and
            # has not already been assigned a B- or I- tag, set its label to "O".
            if token_start != 0 or token_end != 0:
                if labels[i] == -100:
                    labels[i] = 0 # "O" label
        return labels


class NERDataModule:
    """
    A data module to handle loading and preparing data for NER models.
    """

    def __init__(self, config, train_file=None, test_file=None):
        """
        Args:
            config (dict): The training or evaluation configuration dictionary.
            train_file (str, optional): Path to the training data file.
            test_file (str, optional): Path to the test data file.
        """
        self.config = config
        self.train_file = train_file
        self.test_file = test_file
        
        model_path = config.get('model_path') or config.get('model', {}).get('base_model')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.label_map = self._create_label_map()
        self.warned_entities = set()

    def _create_label_map(self):
        """
        Creates a mapping from entity labels to integer IDs using the config file.
        """
        # For evaluation, entity_labels might be in the loaded model's config
        if 'entity_labels' in self.config.get('model', {}):
            entity_labels = self.config['model']['entity_labels']
        else: # Fallback for evaluation config
             entity_labels = ["FIND", "REG", "OBS", "GANGLIOS"] # Provide a default or load from model
        
        label_map = {"O": 0}
        for label in entity_labels:
            label_map[f"B-{label}"] = len(label_map)
            label_map[f"I-{label}"] = len(label_map)
        return label_map

    def setup(self, stage=None):
        """Creates the training and/or test datasets."""
        if self.train_file:
            self.train_dataset = NERDataset(
                file_path=self.train_file,
                tokenizer=self.tokenizer,
                label_map=self.label_map,
                warned_entities_set=self.warned_entities
            )
        if self.test_file:
            self.test_dataset = NERDataset(
                file_path=self.test_file,
                tokenizer=self.tokenizer,
                label_map=self.label_map
            )

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.get('trainer', {}).get('batch_size', 8),
            shuffle=True
        )

    def test_dataloader(self):
        """Returns the DataLoader for the test set."""
        if not hasattr(self, 'test_dataset'):
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.get('batch_size', 16)
        )
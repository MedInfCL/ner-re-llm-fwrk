import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import itertools
import warnings

class REDataset(Dataset):
    """
    A PyTorch Dataset for Relation Extraction.
    Transforms raw text records into individual training instances for every
    possible pair of entities within each record.
    """

    def __init__(self, file_path, tokenizer, relation_map, warned_relations_set=None):
        """
        Args:
            file_path (str): Path to the .jsonl data file.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer for encoding text.
            relation_map (dict): A mapping from relation labels (str) to integer IDs.
            warned_relations_set (set, optional): A set to track relations that have already triggered a warning.
        """
        self.tokenizer = tokenizer
        self.relation_map = relation_map
        self.warned_relations = warned_relations_set if warned_relations_set is not None else set()
        self.instances = self._create_instances(file_path)

    def _create_instances(self, file_path):
        """
        Reads a .jsonl file and creates a flat list of RE instances.
        Each instance represents a pair of entities from a single record.
        """
        instances = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                text = record["text"]
                entities = record.get("entities", [])
                relations = record.get("relations", [])

                # --- Data Validation ---
                for entity in entities:
                    if not isinstance(entity, dict):
                        raise TypeError(f"Entities must be dictionaries, but found: {type(entity)}")
                for rel in relations:
                    if not isinstance(rel, dict):
                        raise TypeError(f"Relations must be dictionaries, but found: {type(rel)}")

                relation_lookup = {(rel["from_id"], rel["to_id"]): rel["type"] for rel in relations}

                for head, tail in itertools.permutations(entities, 2):
                    relation_type = relation_lookup.get((head["id"], tail["id"]), "No_Relation")

                    # If a relation type from the data is not in the config, it is not a valid
                    # instance for training, but "No_Relation" is always considered valid.
                    if relation_type not in self.relation_map:
                        if relation_type not in self.warned_relations:
                            warnings.warn(f"Relation type '{relation_type}' not in config and will be ignored.")
                            self.warned_relations.add(relation_type)
                        continue

                    instances.append({
                        "text": text,
                        "head": head,
                        "tail": tail,
                        "relation": relation_type
                    })
        return instances

    def __len__(self):
        """Returns the total number of entity pair instances."""
        return len(self.instances)

    def __getitem__(self, idx):
        """
        Retrieves, formats, and tokenizes a single RE instance.

        Args:
            idx (int): The index of the instance to retrieve.

        Returns:
            dict: A dictionary containing 'input_ids', 'attention_mask', and 'label'.
        """
        instance = self.instances[idx]
        text = instance["text"]
        head = instance["head"]
        tail = instance["tail"]
        
        # Insert entity markers into the text
        # The order of insertion matters to handle nested or overlapping entities correctly
        head_start, head_end = head['start_offset'], head['end_offset']
        tail_start, tail_end = tail['start_offset'], tail['end_offset']

        # To avoid issues with insertion order, we mark from end to start
        if head_start < tail_start:
            # Head appears first
            text_marked = (
                text[:head_start] + "[E1_START]" + text[head_start:head_end] + "[E1_END]" +
                text[head_end:tail_start] + "[E2_START]" + text[tail_start:tail_end] + "[E2_END]" +
                text[tail_end:]
            )
        else:
            # Tail appears first
            text_marked = (
                text[:tail_start] + "[E2_START]" + text[tail_start:tail_end] + "[E2_END]" +
                text[tail_end:head_start] + "[E1_START]" + text[head_start:head_end] + "[E1_END]" +
                text[head_end:]
            )

        tokenized_inputs = self.tokenizer(
            text_marked,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        label = torch.tensor(self.relation_map[instance["relation"]], dtype=torch.long)

        return {
            "input_ids": tokenized_inputs["input_ids"].squeeze(0),
            "attention_mask": tokenized_inputs["attention_mask"].squeeze(0),
            "label": label,
        }


class REDataModule:
    """
    A data module to handle loading and preparing data for RE models.
    """

    def __init__(self, config, train_file=None, test_file=None):
        """
        Args:
            config (dict): The RE training or evaluation configuration dictionary.
            train_file (str, optional): Path to the training data file.
            test_file (str, optional): Path to the test data file.
        """
        self.config = config
        self.train_file = train_file
        self.test_file = test_file
        
        # Check for model_path during evaluation, fall back to base_model for training
        model_path = config.get('model_path') or config.get('model', {}).get('base_model')
        if not model_path:
            raise ValueError("Could not find 'model_path' or 'model.base_model' in the configuration.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._add_special_tokens()
        
        self.relation_map = self._create_relation_map()
        self.warned_relations = set()

    def _add_special_tokens(self):
        """Adds custom entity marker tokens to the tokenizer."""
        special_tokens_dict = {
            'additional_special_tokens': [
                '[E1_START]', '[E1_END]',  # Head entity
                '[E2_START]', '[E2_END]'   # Tail entity
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)

    def _create_relation_map(self):
        """Creates a mapping from relation labels to integer IDs."""
        relation_labels = self.config['model']['relation_labels']
        return {label: i for i, label in enumerate(relation_labels)}

    def setup(self, stage=None):
        """Creates the training and/or test datasets."""
        if self.train_file:
            self.train_dataset = REDataset(
                file_path=self.train_file,
                tokenizer=self.tokenizer,
                relation_map=self.relation_map,
                warned_relations_set=self.warned_relations
            )
        if self.test_file:
            self.test_dataset = REDataset(
                file_path=self.test_file,
                tokenizer=self.tokenizer,
                relation_map=self.relation_map
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
        
    

# src/evaluation/predictor.py
import torch
from tqdm import tqdm
import numpy as np

class Predictor:
    """
    Handles loading a trained model and running inference on a test dataset.
    This class is designed to be task-agnostic (NER or RE).
    """

    def __init__(self, model, device):
        """
        Initializes the Predictor.

        Args:
            model (torch.nn.Module): The trained model instance (e.g., BertNerModel or REModel).
            device (torch.device): The device to run inference on ('cuda' or 'cpu').
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def predict(self, test_dataloader, task_type):
        """
        Runs inference on the provided dataloader.

        Args:
            test_dataloader (DataLoader): The DataLoader for the test set.
            task_type (str): The type of task, either 'ner' or 're'. This determines
                             how predictions are handled.

        Returns:
            tuple: A tuple containing:
                - all_predictions (list): A list of predicted label sequences/IDs.
                - all_true_labels (list): A list of ground-truth label sequences/IDs.
                - all_logits (numpy.ndarray): An array of the raw output logits from the model.
        """
        all_predictions = []
        all_true_labels = []
        all_logits = []

        progress_bar = tqdm(test_dataloader, desc=f"Evaluating for {task_type.upper()}")
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to the correct device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels' if task_type == 'ner' else 'label'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Get the most likely class prediction
                logits = outputs.logits
                if task_type == 'ner':
                    # For NER, predictions are per-token
                    predictions = torch.argmax(logits, dim=2)
                else: # RE
                    # For RE, predictions are per-sequence
                    predictions = torch.argmax(logits, dim=1)

                # Move data back to CPU for evaluation
                predictions = predictions.detach().cpu().numpy()
                true_labels = labels.detach().cpu().numpy()
                batch_logits = logits.detach().cpu().numpy()

                all_logits.append(batch_logits)

                if task_type == 'ner':
                    # For NER, we need the full prediction sequence for each record
                    # to align with the tokenizer's full offset mapping later.
                    all_predictions.extend(list(predictions))
                    
                    # For calculating metrics, we still need to align true labels.
                    for i in range(len(true_labels)):
                        true_labels_i = []
                        for j in range(len(true_labels[i])):
                            if true_labels[i][j] != -100: # Ignore padded tokens
                                true_labels_i.append(true_labels[i][j])
                        all_true_labels.append(true_labels_i)
                else: # RE
                    # For RE, labels are simpler (one per instance)
                    all_predictions.extend(predictions)
                    all_true_labels.extend(true_labels)

        # If the dataloader was empty, return empty containers.
        if not all_logits:
            return [], [], np.array([])

        # Vertically stack the logits from all batches into a single numpy array
        all_logits = np.concatenate(all_logits, axis=0)

        return all_predictions, all_true_labels, all_logits
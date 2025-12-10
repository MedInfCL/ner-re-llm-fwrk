# src/models/re_model.py
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class REModel(nn.Module):
    """
    A wrapper around a Hugging Face AutoModelForSequenceClassification model,
    adapted for Relation Extraction (RE).
    """

    def __init__(self, base_model, n_labels=None, tokenizer=None):
        """
        Initializes the RE model.

        If `n_labels` is provided (i.e., during training), it configures the model
        with a new sequence classification head and resizes embeddings for the tokenizer.

        If `n_labels` is None (i.e., during evaluation), it loads the model
        directly from the `base_model` path, which is assumed to be a fine-tuned model.

        Args:
            base_model (str): The name or path of the transformer model.
            n_labels (int, optional): The total number of unique relation labels. Defaults to None.
            tokenizer (transformers.PreTrainedTokenizer, optional): The tokenizer, required for training.
        """
        super().__init__()
        self.base_model_name = base_model
        self.n_labels = n_labels

        if n_labels is not None:
            # Training mode: Configure a new model
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided when training a new REModel (n_labels is not None).")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_name,
                num_labels=self.n_labels
            )
            # Resize token embeddings for special entity markers
            self.model.resize_token_embeddings(len(tokenizer))
        else:
            # Evaluation mode: Load a fine-tuned model directly from the path
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_name
            )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Performs the forward pass through the sequence classification model.

        Args:
            input_ids (torch.Tensor): A batch of token IDs, including markers.
            attention_mask (torch.Tensor): The attention mask for the batch.
            labels (torch.Tensor, optional): The ground-truth relation label IDs.

        Returns:
            transformers.modeling_outputs.SequenceClassifierOutput:
                An object containing the loss (if labels are provided) and logits.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def save_pretrained(self, output_dir):
        """
        Saves the underlying Hugging Face model to the specified directory.
        This method delegates the call to the internal model's `save_pretrained`.

        Args:
            output_dir (str): The directory where the model will be saved.
        """
        self.model.save_pretrained(output_dir)
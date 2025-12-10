import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoConfig

class BertNerModel(nn.Module):
    """
    A wrapper around a Hugging Face AutoModelForTokenClassification model.
    This class simplifies the model initialization and forward pass for NER tasks.
    """

    def __init__(self, base_model, n_labels=None):
        """
        Initializes the NER model.

        If `n_labels` is provided (i.e., during training), it configures the model
        with a new token classification head.

        If `n_labels` is None (i.e., during evaluation), it loads the model
        directly from the `base_model` path, which is assumed to be a fine-tuned model.

        Args:
            base_model (str): The name or path of the transformer model.
            n_labels (int, optional): The total number of unique labels. Defaults to None.
        """
        super().__init__()
        self.base_model_name = base_model
        self.n_labels = n_labels

        if n_labels is not None:
            # Training mode: Configure a new model with a classification head
            config = AutoConfig.from_pretrained(
                self.base_model_name,
                num_labels=self.n_labels
            )
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.base_model_name,
                config=config
            )
        else:
            # Evaluation mode: Load a fine-tuned model directly from the path
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.base_model_name
            )


    def forward(self, input_ids, attention_mask, labels=None):
        """
        Performs the forward pass through the model.

        Args:
            input_ids (torch.Tensor): A batch of token IDs.
            attention_mask (torch.Tensor): The attention mask for the batch.
            labels (torch.Tensor, optional): The ground-truth label IDs.

        Returns:
            transformers.modeling_outputs.TokenClassifierOutput:
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

        Args:
            output_dir (str): The directory where the model will be saved.
        """
        self.model.save_pretrained(output_dir)
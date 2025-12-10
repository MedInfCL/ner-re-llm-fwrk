import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os

class Trainer:
    """
    A class to handle the training and evaluation of a Hugging Face model for NER.
    """

    def __init__(self, model, datamodule, config):
        """
        Initializes the Trainer.

        Args:
            model (torch.nn.Module): The model to be trained.
            datamodule (NERDataModule): The data module providing the dataloaders.
            config (dict): The configuration dictionary containing trainer settings.
        """
        self.model = model
        self.datamodule = datamodule
        self.config = config
        self.device = torch.device(config['trainer']['device'] if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        """
        Executes the full training loop for the model.
        """
        train_dataloader = self.datamodule.train_dataloader()

        # Handle an empty dataloader
        if len(train_dataloader) == 0:
            print("Warning: The training dataloader is empty. Skipping training.")
            return
        
        # Prepare optimizer and scheduler
        optimizer = self._create_optimizer()
        total_steps = len(train_dataloader) * self.config['trainer']['n_epochs']
        scheduler = self._create_scheduler(optimizer, total_steps)

        print("Starting training...")
        for epoch in range(self.config['trainer']['n_epochs']):
            print(f"\n--- Epoch {epoch + 1}/{self.config['trainer']['n_epochs']} ---")
            
            self.model.train()
            total_loss = 0
            
            # Progress bar for the current epoch
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} Training")

            for batch in progress_bar:
                optimizer.zero_grad()

                # Move batch to the correct device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                label_key = 'labels' if 'labels' in batch else 'label'
                labels = batch[label_key].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Average training loss for epoch {epoch + 1}: {avg_train_loss:.4f}")

        print("\nTraining complete.")
        
    def _create_optimizer(self):
        """Creates and returns the AdamW optimizer."""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['trainer']['weight_decay'],
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        learning_rate = float(self.config['trainer']['learning_rate'])
        return AdamW(optimizer_grouped_parameters, lr=learning_rate)

    def _create_scheduler(self, optimizer, total_steps):
        """Creates and returns the learning rate scheduler."""
        # Handle case where there are no steps
        if total_steps == 0:
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

        warmup_steps = int(total_steps * self.config['trainer']['warmup_ratio'])
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
    def save_model(self, output_dir):
        """
        Saves the trained model and tokenizer to the specified directory.
        
        Args:
            output_dir (str): The directory where the model will be saved.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.model.save_pretrained(output_dir)
        self.datamodule.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
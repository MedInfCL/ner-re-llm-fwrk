# tests/unit/src/training/test_trainer.py
import pytest
import torch
import yaml
from unittest.mock import MagicMock, patch

# Add the project root to the Python path to allow for absolute imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.training.trainer import Trainer

# Mock configuration for testing
@pytest.fixture
def mock_config():
    """Provides a mock configuration dictionary for the trainer."""
    return {
        'trainer': {
            'device': 'cpu',
            'n_epochs': 1,
            'weight_decay': 0.01,
            'learning_rate': 5e-5,
            'warmup_ratio': 0.1,
        },
        'paths': {
            'output_dir': 'mock_output'
        }
    }

# Mock model with necessary attributes and methods
@pytest.fixture
def mock_model():
    """Creates a mock model that simulates a Hugging Face model."""
    model = MagicMock()
    model.to.return_value = model
    model.train.return_value = None
    
    # Mock the forward pass to return a loss
    mock_loss = MagicMock()
    mock_loss.item.return_value = 0.5
    mock_loss.backward.return_value = None
    model.return_value = MagicMock(loss=mock_loss)
    
    # Mock parameter methods
    model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
    model.named_parameters.return_value = [('param', torch.nn.Parameter(torch.randn(2, 2)))]
    
    # Mock the save_pretrained method on the model wrapper itself
    model.save_pretrained.return_value = None
    
    return model

# Mock data module with a mock dataloader
@pytest.fixture
def mock_datamodule():
    """Creates a mock data module with a simplified dataloader."""
    datamodule = MagicMock()
    
    # The batch that the dataloader should yield
    mock_batch = {
        'input_ids': torch.ones((1, 10), dtype=torch.long),
        'attention_mask': torch.ones((1, 10), dtype=torch.long),
        'labels': torch.ones((1, 10), dtype=torch.long),
    }
    
    # Create a mock loader that behaves like a real DataLoader by
    # implementing __len__ and __iter__
    mock_loader = MagicMock()
    mock_loader.__iter__.return_value = iter([mock_batch])
    mock_loader.__len__.return_value = 1
    
    datamodule.train_dataloader.return_value = mock_loader
    
    # Mock tokenizer for saving
    datamodule.tokenizer = MagicMock()
    datamodule.tokenizer.save_pretrained.return_value = None
    
    return datamodule

# Test case for trainer initialization
def test_trainer_initialization(mock_model, mock_datamodule, mock_config):
    """
    Tests if the Trainer initializes correctly by setting the model, datamodule,
    config, and moving the model to the specified device.
    """
    trainer = Trainer(model=mock_model, datamodule=mock_datamodule, config=mock_config)
    
    assert trainer.model is mock_model
    assert trainer.datamodule is mock_datamodule
    assert trainer.config is mock_config
    assert trainer.device == torch.device('cpu')
    mock_model.to.assert_called_once_with(torch.device('cpu'))

# Test case for the optimizer creation
def test_create_optimizer(mock_model, mock_datamodule, mock_config):
    """
    Tests the _create_optimizer method to ensure it returns a valid AdamW optimizer
    with the correct learning rate.
    """
    trainer = Trainer(model=mock_model, datamodule=mock_datamodule, config=mock_config)
    optimizer = trainer._create_optimizer()
    
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == float(mock_config['trainer']['learning_rate'])

# Test case for the scheduler creation
def test_create_scheduler(mock_model, mock_datamodule, mock_config):
    """
    Tests the _create_scheduler method to ensure it returns a valid learning
    rate scheduler.
    """
    trainer = Trainer(model=mock_model, datamodule=mock_datamodule, config=mock_config)
    optimizer = trainer._create_optimizer()
    total_steps = len(mock_datamodule.train_dataloader()) * mock_config['trainer']['n_epochs']
    
    scheduler = trainer._create_scheduler(optimizer, total_steps)
    
    # Verify that the correct type of scheduler was created
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

# Test case for the main training loop
@patch('src.training.trainer.tqdm', side_effect=lambda iterable, **kwargs: iterable) # Mock tqdm to return the original iterable
def test_train_loop(mock_tqdm, mock_model, mock_datamodule, mock_config):
    """
    Tests the main training loop to ensure it runs without errors and
    performs the expected calls (forward, backward, step).
    """
    trainer = Trainer(model=mock_model, datamodule=mock_datamodule, config=mock_config)
    
    optimizer = MagicMock()
    scheduler = MagicMock()
    trainer._create_optimizer = MagicMock(return_value=optimizer)
    trainer._create_scheduler = MagicMock(return_value=scheduler)

    trainer.train()

    mock_model.train.assert_called_once()
    
    # Verify that the core optimizer and scheduler methods were called
    optimizer.zero_grad.assert_called_once()
    mock_model.return_value.loss.backward.assert_called_once()
    optimizer.step.assert_called_once()
    scheduler.step.assert_called_once()

# Test case for the model saving functionality
@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_save_model(mock_makedirs, mock_exists, mock_model, mock_datamodule, mock_config):
    """
    Tests the save_model method to ensure it calls the save_pretrained methods
    on both the model and the tokenizer, and that it creates the directory.
    """
    trainer = Trainer(model=mock_model, datamodule=mock_datamodule, config=mock_config)
    output_dir = "test_output/model"
    
    trainer.save_model(output_dir)
    
    # Verify the check for the directory's existence
    mock_exists.assert_called_once_with(output_dir)
    
    # Verify directory creation was attempted correctly
    mock_makedirs.assert_called_once_with(output_dir)
    
    # Verify that the save methods were called correctly
    mock_model.save_pretrained.assert_called_once_with(output_dir)
    mock_datamodule.tokenizer.save_pretrained.assert_called_once_with(output_dir)


# Fixture for a datamodule with an empty dataloader
@pytest.fixture
def mock_datamodule_empty():
    """Creates a mock data module with an empty dataloader."""
    datamodule = MagicMock()
    
    # Create a mock loader that is empty
    mock_loader = MagicMock()
    mock_loader.__iter__.return_value = iter([])
    mock_loader.__len__.return_value = 0
    
    datamodule.train_dataloader.return_value = mock_loader
    
    return datamodule

# Test case for training with an empty dataloader
def test_train_with_empty_dataloader(mock_model, mock_datamodule_empty, mock_config):
    """
    Tests that the training loop handles an empty dataloader gracefully
    and does not perform any training steps.
    """
    trainer = Trainer(model=mock_model, datamodule=mock_datamodule_empty, config=mock_config)
    
    # Mock the optimizer and scheduler to check if they are called
    optimizer = MagicMock()
    scheduler = MagicMock()
    trainer._create_optimizer = MagicMock(return_value=optimizer)
    trainer._create_scheduler = MagicMock(return_value=scheduler)

    # Run the training process
    trainer.train()

    # Assert that no training steps were taken
    optimizer.step.assert_not_called()
    scheduler.step.assert_not_called()

# Test case for gradient clipping
@patch('torch.nn.utils.clip_grad_norm_')
def test_gradient_clipping_is_called(mock_clip_grad_norm, mock_model, mock_datamodule, mock_config):
    """
    Tests that gradient clipping is called during the training loop.
    """
    trainer = Trainer(model=mock_model, datamodule=mock_datamodule, config=mock_config)

    # Run the training process for one step
    trainer.train()

    # Assert that clip_grad_norm_ was called once
    mock_clip_grad_norm.assert_called_once()
    
    # Assert that it was called with the model's parameters and a clipping value of 1.0
    mock_clip_grad_norm.assert_called_with(mock_model.parameters(), 1.0)

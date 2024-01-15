import os
import json
import torch
from dataclasses import dataclass
from typing import Any

@dataclass
class Trainer:
    """Main class for model training. Handles the training and validation processes, checkpointing, and saving the final model."""
    model: torch.nn.Module  # PyTorch model to be trained
    epochs: int  # Number of training epochs
    train_dataloader: torch.utils.data.DataLoader  # DataLoader for training data
    val_dataloader: torch.utils.data.DataLoader  # DataLoader for validation data
    criterion: Any  # Loss function
    optimizer: Any  # Optimizer for training
    lr_scheduler: Any  # Learning rate scheduler
    device: str  # Device to train the model on (e.g., 'cuda' or 'cpu')
    model_dir: str  # Directory to save model checkpoints
    model_name: str  # Name of the model

    def __post_init__(self):
        # Initialize training and validation loss storage
        self.loss = {"train": [], "val": []}
        # Move model to the specified device
        self.model.to(self.device)
    
    def train(self):
        """Conducts the training and validation process across all epochs."""
        for epoch in range(self.epochs):
            # Train for one epoch
            self._train_epoch()
            # Validate for one epoch
            self._validate_epoch()
            print(f"Epoch: {epoch + 1}/{self.epochs}, Train Loss={self.loss['train'][-1]:.5f}, Val Loss={self.loss['val'][-1]:.5f}")

            # Step the learning rate scheduler
            self.lr_scheduler.step()

    
    def _train_epoch(self):
        self.model.train()
        running_loss = 0

        for batch_data in self.train_dataloader:
            src, tgt = batch_data # [batch, seq_length]
            src, tgt = src.to(self.device), tgt.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(src, tgt[:, :-1], self.device)  # tgt[:, :-1] to shift for prediction

            # Compute loss
            loss = self.criterion(output.contiguous().view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
            running_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

        # Compute average loss for the epoch
        avg_loss = running_loss / len(self.train_dataloader)
        self.loss['train'].append(avg_loss)

    def _validate_epoch(self):
        self.model.eval()
        running_loss = 0

        with torch.no_grad():
            for batch_data in self.val_dataloader:
                src, tgt = batch_data
                src, tgt = src.to(self.device), tgt.to(self.device)

                output = self.model(src, tgt[:, :-1], self.device)  # Shift for prediction in validation

                # Compute loss
                loss = self.criterion(output.contiguous().view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
                running_loss += loss.item()
            avg_loss = running_loss / len(self.val_dataloader)
            self.loss['val'].append(avg_loss)

    def save_model(self):
        """Saves the final trained model."""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Saves the training and validation loss history."""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)  
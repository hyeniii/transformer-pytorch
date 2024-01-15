import argparse
import yaml
import os 
import torch
import torch.nn as nn
import torch.optim as optim
import src.transformer as t
from src.dataloader import get_dataloader_and_vocab
from src.trainer import Trainer 
from torch.optim.lr_scheduler import LambdaLR

def train(config):
    # Create necessary directories for saving model artifacts
    if not os.path.exists(config['model_dir']):
        os.makedirs(config['model_dir'])

    # Load training data
    train_dataloader, tokenizer = get_dataloader_and_vocab(
        ds_type = "train",
        batch_size = config['train_batch_size'],
        shuffle = config['shuffle']
    ) #[batch, seq_length]

    # Load validation data
    val_dataloader, _ = get_dataloader_and_vocab(
        ds_type = "valid",
        batch_size = config['val_batch_size'],
        shuffle = config['shuffle']
    )
    # Determine the vocabulary size
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # Initialize the model, loss criterion, optimizer, and learning rate scheduler
    model = t.Transformer(config["src_vocab_size"], 
                          config["tgt_vocab_size"], 
                          config["d_model"], 
                          config["num_heads"],
                          config["num_layers"],
                          config["d_ff"],
                          config["max_seq_length"],
                          config["dropout"]
                            )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), 
                        lr=0.0001, 
                        betas = (0.9, 0.98),
                        eps=1e-9)
    total_epochs = config['epochs']
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)
    # Set the device (GPU or CPU) for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the trainer with all the training components
    trainer = Trainer(
        model = model,
        epochs = config['epochs'],
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        criterion = criterion,
        optimizer = optimizer,
        lr_scheduler = lr_scheduler,
        device = device,
        model_dir = config['model_dir'],
        model_name = config['model_name'],
    )
    
    # Start the training process
    trainer.train()
    print("Training finished")

    # Save the model, loss history, vocabulary, and configuration
    trainer.save_model()
    trainer.save_loss()
    save_vocab(tokenizer, config['model_dir'])
    save_config(config, config['model_dir'])
    print("Model artifacts saved to folder:", config['model_dir'])

def save_config(config:dict, model_dir:str):
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def save_vocab(vocab, model_dir:str):
    vocab_path = os.path.join(model_dir, 'vocab.pt')
    with open(vocab_path, "wb") as f:
        torch.save(vocab, f)

if __name__ == '__main__':
    # Parse arguments for configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, default='./config.yaml', help='path to yaml config')
    args = parser.parse_args()

    # Load configuration from the yaml file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train(config)
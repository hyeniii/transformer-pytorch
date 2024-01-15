from datasets import load_dataset
from transformers import MBartTokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from functools import partial

def load_dataset_and_tokenizer(language_pair="en-fr"):
    # Load the dataset
    dataset = load_dataset("opus_books", language_pair)

    reduced_dataset = dataset["train"].select(range(1000))

    # Split the training dataset into training and validation sets
    train_test_split = reduced_dataset.train_test_split(test_size=0.2, seed=42)
    dataset["train"] = train_test_split["train"]
    dataset["valid"] = train_test_split["test"]

    # Initialize the tokenizer
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")

    return dataset, tokenizer

def collate_batch(batch, tokenizer, max_length=512):
    src_list, tgt_list = [], []
    for item in batch:
        translation = item["translation"]
        src_sample, tgt_sample = translation["en"], translation["fr"]
        # Tokenize and encode the source and target text
        src_encoded = tokenizer.encode(src_sample, add_special_tokens=True, truncation=True, max_length=max_length)
        tgt_encoded = tokenizer.encode(tgt_sample, add_special_tokens=True, truncation=True, max_length=max_length)

        src_list.append(torch.tensor(src_encoded, dtype=torch.long))
        tgt_list.append(torch.tensor(tgt_encoded, dtype=torch.long))

    # Pad sequences to the maximum length in this batch
    src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=tokenizer.pad_token_id)

    return src_padded, tgt_padded


def get_dataloader_and_vocab(ds_type, batch_size, shuffle):
    # Load dataset and tokenizer
    dataset, tokenizer = load_dataset_and_tokenizer()

    # Define collate function for DataLoader
    collate_fn = partial(collate_batch, tokenizer=tokenizer)

    # Create DataLoader
    dataloader = DataLoader(dataset[ds_type], batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return dataloader, tokenizer
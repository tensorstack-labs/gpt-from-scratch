"""
03_sliding_window_dataset.py
TensorStack - Chapter 2: Sliding Window Dataset & PyTorch DataLoader

Implements:
1. GPTDatasetV1: Custom torch.utils.data.Dataset for autoregressive token modeling.
2. create_dataloader_v1: Factory function returning batched, shuffled PyTorch DataLoaders.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from typing import Tuple


class GPTDatasetV1(Dataset):
    """
    Extracts fixed-length overlapping sequence chunks using a sliding window.
    For each input chunk of length `max_length`, creates a target chunk shifted right by +1 token.
    """
    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        # Tokenize entire text corpus
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Apply sliding window across token stream
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Creates a high-performance PyTorch DataLoader ready for GPU training."""
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length=max_length, stride=stride)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    sample_text = (
        "In the beginning God created the heavens and the earth. Now the earth was "
        "formless and empty, darkness was over the surface of the deep, and the Spirit of God "
        "was hovering over the waters. And God said, 'Let there be light,' and there was light. "
        "God saw that the light was good, and he separated the light from the darkness."
    )
    
    print("==================================================")
    print("🚀 TENSORSTACK: PYTORCH DATALOADER PIPELINE DEMO")
    print("==================================================")
    
    batch_size = 2
    max_length = 4
    stride = 1
    
    dataloader = create_dataloader_v1(
        sample_text,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False,
        drop_last=False
    )
    
    print(f"✔ Created DataLoader with {len(dataloader.dataset)} total sequence pairs.")
    print(f"✔ Batch Size: {batch_size} | Sequence Length (Context): {max_length}\n")
    
    tokenizer = tiktoken.get_encoding("gpt2")
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    
    print("📦 FIRST BATCH INSPECTION:")
    print(f"  Inputs Shape:  {inputs.shape}   # [Batch_Size, Context_Length]")
    print(f"  Targets Shape: {targets.shape}  # [Batch_Size, Context_Length]\n")
    
    for b in range(inputs.shape[0]):
        in_tokens = [tokenizer.decode([i.item()]) for i in inputs[b]]
        tgt_tokens = [tokenizer.decode([i.item()]) for i in targets[b]]
        print(f"  Sample {b} Input  (x): {inputs[b].tolist()} -> {in_tokens}")
        print(f"  Sample {b} Target (y): {targets[b].tolist()} -> {tgt_tokens}")
        print(f"  Notice Target is shifted by +1 token!\n")
    print("==================================================")

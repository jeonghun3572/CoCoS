import json

import torch


class Collator:
    """Pass-through collator that returns the batch as-is."""

    def __call__(self, batch):
        return batch


class Dataset(torch.utils.data.Dataset):
    """Simple dataset that loads JSON/JSONL files."""

    def __init__(self, path):
        self.data = self._load_data(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _load_data(self, path):
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise ValueError(f"Unsupported file format: {path}")

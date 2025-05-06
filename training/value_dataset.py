import numpy as np
import torch
from torch.utils.data import Dataset

class ValueDataset(Dataset):
    def __init__(self, hole_cards, round_states, values):
        self.hole_cards = hole_cards
        self.round_states = round_states
        self.values = torch.tensor(values, dtype=torch.float32)

    def __len__(self):
        return len(self.hole_cards)

    def __getitem__(self, idx):
        return self.hole_cards[idx], self.round_states[idx], self.values[idx]
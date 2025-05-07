import numpy as np
import torch
from torch.utils.data import Dataset

class ValueDataset(Dataset):
    def __init__(self, hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes, values):
        self.hole_suit = torch.tensor(hole_suit, dtype=torch.int64)
        self.hole_rank = torch.tensor(hole_rank, dtype=torch.int64)
        self.hole_card_idx = torch.tensor(hole_card_idx, dtype=torch.int64)
        self.board_suit = torch.tensor(board_suit, dtype=torch.int64)
        self.board_rank = torch.tensor(board_rank, dtype=torch.int64)
        self.board_card_idx = torch.tensor(board_card_idx, dtype=torch.int64)
        self.actions_occured = torch.tensor(actions_occured, dtype=torch.int64)
        self.bet_sizes = torch.tensor(bet_sizes, dtype=torch.float32)
        self.values = torch.tensor(values, dtype=torch.float32)

    def __len__(self):
        return len(self.hole_suit)

    def __getitem__(self, idx):

        return self.hole_suit[idx], self.hole_rank[idx], self.hole_card_idx[idx], \
                self.board_suit[idx], self.board_rank[idx], self.board_card_idx[idx], \
                self.actions_occured[idx], self.bet_sizes[idx], self.values[idx]
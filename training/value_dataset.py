import numpy as np
import torch
from torch.utils.data import Dataset

class ValueDataset(Dataset):
    def __init__(self, hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes, action, values):
        self.hole_suit = hole_suit
        self.hole_rank = hole_rank
        self.hole_card_idx = hole_card_idx
        self.board_suit = board_suit
        self.board_rank = board_rank
        self.board_card_idx = board_card_idx
        self.actions_occured = actions_occured
        self.bet_sizes = bet_sizes
        self.action = action
        self.values = values
    def __len__(self):
        return len(self.hole_suit)

    def __getitem__(self, idx):

        return self.hole_suit[idx], self.hole_rank[idx], self.hole_card_idx[idx], \
                self.board_suit[idx], self.board_rank[idx], self.board_card_idx[idx], \
                self.actions_occured[idx], self.bet_sizes[idx], self.action[idx], self.values[idx]
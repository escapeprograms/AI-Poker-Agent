import torch
import torch.nn as nn
import numpy as np
from utils import SUITS, RANKS
from encode_state import encode_card


class PolicyNet(nn.Module):
    def __init__(self, embedding_dim=32):
        super(PolicyNet, self).__init__()

        # Define the suits and ranks
        self.suits = SUITS  
        self.ranks = RANKS

        self.num_suits = len(self.suits)
        self.num_ranks = len(self.ranks)
        self.embedding_dim = embedding_dim

        # Embeddings for suit, rank, and individual card
        self.suit_embedding = nn.Embedding(self.num_suits+1, embedding_dim)
        self.rank_embedding = nn.Embedding(self.num_ranks+1, embedding_dim)
        self.card_embedding = nn.Embedding(self.num_ranks*self.num_suits+1, embedding_dim)


        # Create mappings from card strings to indices
        self.suit_to_index = {suit: i for i, suit in enumerate(self.suits)}
        self.rank_to_index = {rank: i for i, rank in enumerate(self.ranks)}

        # Create layers
        self.fc1 = nn.Linear(embedding_dim+2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3) # fold, call, raise
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, suits, ranks, card_ids, pot, canRaise):
        suit_indices = torch.tensor(suits, dtype=torch.int)
        rank_indices = torch.tensor(ranks, dtype=torch.int)
        card_indices = torch.tensor(card_ids, dtype=torch.int)

        suit_embedding = self.suit_embedding(suit_indices)
        rank_embedding = self.rank_embedding(rank_indices)
        card_embedding = self.card_embedding(card_indices)
        
        final_embeddings = suit_embedding + rank_embedding + card_embedding # Embeddings of each card in a 'list'
        hand_embeddings = torch.sum(final_embeddings, dim=0) # Create an embedding for the hand by summing embeddings for the cards

        x = torch.cat((hand_embeddings, torch.tensor((pot, canRaise))))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        # Note that x is already nonnegative because of sigmoid
        x = x / x.sum() # Normalize so sum is 1
        return x

def model_wrapper(model: PolicyNet):
    def policy(hand: list, community: list, pot: int, canRaise: bool):
        cards = hand+community
        model.eval()
        suits, ranks, cardIds = [], [], []
        for card in cards:
            s, r, c = encode_card(card)
            suits.append(s)
            ranks.append(r)
            cardIds.append(c)
        output = model(suits, ranks, cardIds, pot, int(canRaise))
        # We assume that the output is already nonnegative!
        if not canRaise:
            output[2] = 0
        output = output / output.sum()
        return output.detach().numpy()#output.tolist()
    return policy

CURR_MODEL = PolicyNet()
#CURR_MODEL.load_state_dict(torch.load("models/policy[v3].pth"))

CURR_MODEL.load_state_dict(torch.load("CURR_POLICY.pth"))

# CURR_MODEL.load_state_dict(torch.load("models/evaluation_function[latest-test].pth"))
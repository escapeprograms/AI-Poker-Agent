import pprint

import torch

from CFRD_player import CFRDPlayer
from value_model import ValueNetwork

import torch
import os
import sys

pp = pprint.PrettyPrinter(indent=2)

class CustomPlayer(CFRDPlayer):
    evaluation_function = ValueNetwork()
    model_path = os.path.join(os.path.dirname(__file__), "models/CFR-D_cp0_fixed.pth")
    evaluation_function.load_state_dict(torch.load(model_path, weights_only=True))  
    def __init__(self):
        super().__init__(self.evaluation_function, device="cpu")

def setup_ai():
    return CustomPlayer()

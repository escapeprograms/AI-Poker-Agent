import pprint
import random

import torch

from pypokerengine.players import BasePokerPlayer
from search import minimax, manual_walk
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.utils.game_state_utils import attach_hole_card, attach_hole_card_from_deck, restore_game_state
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.round_manager import RoundManager
from pypokerengine.engine.action_checker import ActionChecker
from encode_state import encode_game_state

import numpy as np
from math import inf

from CFRD_player import CFRDPlayer
from value_model import ValueNetwork

pp = pprint.PrettyPrinter(indent=2)

class CustomPlayer(CFRDPlayer):
    evaluation_function = ValueNetwork()
    evaluation_function.load_state_dict(torch.load("models/good_models/CFR-D1.pth", weights_only=True))  
    def __init__(self):
        super().__init__(evaluation_function=self.evaluation_function, device="cpu")

def setup_ai():
    return CustomPlayer()

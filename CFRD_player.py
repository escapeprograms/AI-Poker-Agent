import pprint
import random

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

pp = pprint.PrettyPrinter(indent=2)

class CFRDPlayer(BasePokerPlayer):       
    def __init__(self, value_network, device='cuda'):
        super().__init__()
        self.value_network = value_network
        self.device = device


    def declare_action(self, valid_actions, hole_card, round_state):


        # Get agent's UUID
        seat = round_state["next_player"]
        uuid = round_state["seats"][seat]["uuid"]
        # Get the game state
        game_state = restore_game_state(round_state)        

        #copied from train_value_model.py simulation
        policy = []
        pred_vals = []
        for action in valid_actions:
            #calculate the value of each action
            val = self.value_network(*encode_game_state(hole_card, game_state, action, seat, device=self.device))
            pred_vals.append(val.item())
        total_val = sum(pred_vals)
        for val in pred_vals:
            if total_val == 0:
                probability = 1 / len(pred_vals) #uniform distribution if no info yet
            else:
                probability = val / total_val
            policy.append(probability)
        
        #choose an action to take
        action_indices = list(range(len(policy)))
        selected_action = random.choices(action_indices, weights=policy, k=1)[0]
        return valid_actions[selected_action]["action"]

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def setup_ai():
    return CFRDPlayer()

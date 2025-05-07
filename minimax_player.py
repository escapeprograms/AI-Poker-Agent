import pprint

from pypokerengine.players import BasePokerPlayer
from search import minimax, manual_walk
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.utils.game_state_utils import attach_hole_card, attach_hole_card_from_deck, restore_game_state
from encode_state import encode_game_state

import numpy as np

pp = pprint.PrettyPrinter(indent=2)

class MinimaxPlayer(BasePokerPlayer):       
    def __init__(self, value_network, train=False):
        super().__init__()
        self.value_network = value_network

        #after each round, store some training data from the round
        #hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes
        self.train_embedded_state = [[] for i in range(8)] #store embedded state, each subarray is a different part of the state
        self.train_values = [] #store the value of the round

        self.cur_round_states = [] #keep previous round states to use for training
        self.hole_card = [] #save hole card

    def declare_action(self, valid_actions, hole_card, round_state):
        self.cur_round_states.append(round_state) #add round state to training list
        self.hole_card = hole_card #save hole card

        # Get agent's UUID
        seat = round_state["next_player"]
        uuid = round_state["seats"][seat]["uuid"]

        # Clone known game state
        game_state = restore_game_state(round_state)            
        game_state = attach_hole_card(game_state, uuid, gen_cards(hole_card))
        # print("gamestate", game_state)
        # Remove hole cards from deck
        for card in gen_cards(hole_card):
            game_state["table"].deck.deck.remove(card)

        # Generate random hole cards for opponents
        for seat in round_state["seats"]:
            if seat["uuid"] != uuid:
                game_state = attach_hole_card_from_deck(game_state, seat["uuid"])

        # Search for best action
        if (np.random.rand() < 0.8):
            return minimax(game_state, None, 2, is_max=True, value_network=self.value_network)[1]["action"]
        else:
            return "call" #randomly call

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state):
        #after each round, store training data containing the hole cards, a partial state, and the value (money won/lost)
        value = round_state['pot']['main']['amount'] * (1 if winners[0]['uuid'] == self.uuid else -1)
        # print("VALUE", value, "WINNER", winners[0]['uuid'])
        # Store multiple training data
        for state in self.cur_round_states:
            #store each aspect of the state
            hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes = encode_game_state(self.hole_card, state)
            self.train_embedded_state[0].append(hole_suit)
            self.train_embedded_state[1].append(hole_rank)
            self.train_embedded_state[2].append(hole_card_idx)
            self.train_embedded_state[3].append(board_suit)
            self.train_embedded_state[4].append(board_rank)
            self.train_embedded_state[5].append(board_card_idx)
            self.train_embedded_state[6].append(actions_occured)
            self.train_embedded_state[7].append(bet_sizes)

            self.train_values.append(value)
            
        self.cur_round_states = [] #reset stored round states
        self.hole_card = []

def setup_ai():
    return MinimaxPlayer()

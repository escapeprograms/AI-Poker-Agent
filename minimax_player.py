import pprint

from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.utils.game_state_utils import attach_hole_card, attach_hole_card_from_deck, restore_game_state
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.round_manager import RoundManager
from pypokerengine.engine.action_checker import ActionChecker
from encode_state import encode_game_state

import numpy as np
from math import inf

pp = pprint.PrettyPrinter(indent=2)

class MinimaxPlayer(BasePokerPlayer):       
    def __init__(self, value_network, train=False):
        super().__init__()
        self.value_network = value_network

        self.pot = 0 #store the pot of the round

        #after each round, store some training data from the round
        #hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes
        self.train_embedded_state = [[] for i in range(8)] #store embedded state, each subarray is a different part of the state
        self.train_values = [] #store the value of the round

        self.cur_round_states = [] #keep previous round states to use for training
        self.hole_card = [] #save hole card

    
    def minimax(self, game_state, events, depth, is_max, value_network):
        max_multiplier = 1 if is_max else -1
        #game is over
        if (game_state["street"] == Const.Street.FINISHED):
            winners = events[-1][1]["message"]["winners"]
            for winner in winners:
                if winner["uuid"] == self.uuid:
                    return max_multiplier, None
            return max_multiplier, None

        #maximum depth reached
        if depth == 0:
            # events[-1][1]["message"]["winners"] stores winner
            round_state = events[-1][1]["message"]["round_state"]
            # print("round state:", round_state)
            hole_card = game_state['table'].seats.players[0 if is_max else 1].hole_card
            val = value_network(*encode_game_state(hole_card, round_state))
            # print(val, str(hole_card[0]), str(hole_card[1]))
            return val * max_multiplier, None
        

        # Generate legal actions at current state
        actions = ActionChecker.legal_actions(
            game_state["table"].seats.players, 
            game_state["next_player"],
            game_state["small_blind_amount"],
            game_state["street"]
        )

        #test: remove fold from actions
        # actions = [action for action in actions if action["action"] != "fold"]
        # Search actions recursively
        top_score = -inf if is_max else inf
        top_action = None
        for action in actions:
            next_state, events = RoundManager.apply_action(game_state, action["action"])
            score, _ = self.minimax(next_state, events, depth - 1, not is_max, value_network)
            if (is_max and score > top_score) or (not is_max and score < top_score):
                top_score, top_action = score, action

            # if depth==2:
            #     print(score,action, is_max)
        return top_score, top_action

    def declare_action(self, valid_actions, hole_card, round_state):
        self.cur_round_states.append(round_state) #add round state to training list
        self.hole_card = hole_card #save hole card
        # print("HOLE CARD", hole_card)

        self.pot = round_state['pot']['main']['amount'] #store the pot of the round
        
        #always call first round
        if round_state['street'] == 'preflop':
            # print("CALLING PREFLOP")
            return "call"

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
        return self.minimax(game_state, None, 2, is_max=True, value_network=self.value_network)[1]["action"]


    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state):
        #after each round, store training data containing the hole cards, a partial state, and the value (money won/lost)
        value = (1 if winners[0]['uuid'] == self.uuid else -1)
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

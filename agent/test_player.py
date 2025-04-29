import pprint

from pypokerengine.players import BasePokerPlayer
from agent.search import minimax
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.utils.game_state_utils import attach_hole_card, attach_hole_card_from_deck, restore_game_state

class TestPlayer(BasePokerPlayer):       
    def receive_game_start_message(self, game_info): 
        # Store game rules
        self.player_num = game_info["player_num"]
        self.max_round = game_info["rule"]["max_round"]
        self.small_blind_amount = game_info["rule"]["small_blind_amount"]
        self.ante = game_info["rule"]["ante"]

    def declare_action(self, valid_actions, hole_card, round_state):
        # Create emulator for search
        emulator = Emulator()
        emulator.set_game_rule(
            self.player_num,
            self.max_round,
            self.small_blind_amount,
            self.ante
        )        
        
        # Get agent's UUID
        seat = round_state["next_player"]
        uuid = round_state["seats"][seat]["uuid"]

        # Clone game state
        game_state = restore_game_state(round_state)            
        game_state = attach_hole_card(game_state, uuid, gen_cards(hole_card))
        for seat in round_state["seats"]:
            if seat["uuid"] != uuid:
                game_state = attach_hole_card_from_deck(game_state, seat["uuid"])

        return minimax(emulator, game_state, uuid, 2, True)

    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def setup_ai():
    return TestPlayer()

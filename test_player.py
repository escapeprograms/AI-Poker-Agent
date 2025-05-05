import pprint

from pypokerengine.players import BasePokerPlayer
from search import minimax, manual_walk
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.utils.game_state_utils import attach_hole_card, attach_hole_card_from_deck, restore_game_state

pp = pprint.PrettyPrinter(indent=2)

class TestPlayer(BasePokerPlayer):       

    def declare_action(self, valid_actions, hole_card, round_state):
        # Get agent's UUID
        seat = round_state["next_player"]
        uuid = round_state["seats"][seat]["uuid"]

        # TODO: use cheat deck in table constructor to control which cards get dealt?

        # Clone known game state
        game_state = restore_game_state(round_state)            
        game_state = attach_hole_card(game_state, uuid, gen_cards(hole_card))

        # Generate random hole cards for opponents
        for seat in round_state["seats"]:
            if seat["uuid"] != uuid:
                game_state = attach_hole_card_from_deck(game_state, seat["uuid"])

        # Search for bes action
        return minimax(game_state, None, 2, True)[1]["action"]

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def setup_ai():
    return TestPlayer()

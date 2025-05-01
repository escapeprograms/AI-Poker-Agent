import pprint
from search import State, minimax, apply_action, legal_actions
from pypokerengine.players import BasePokerPlayer

pp = pprint.PrettyPrinter(indent=2)

class TestPlayer(BasePokerPlayer):       
    def declare_action(self, valid_actions, hole_card, round_state):    
        pp.pprint(round_state)

        print()

        pp.pprint(hole_card)

        state = State(hole_card, round_state)

        print()
        print(state)

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def setup_ai():
    return TestPlayer()

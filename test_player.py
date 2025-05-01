from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import pprint
from search import State, minimax
from pypokerengine.players import BasePokerPlayer

pp = pprint.PrettyPrinter(indent=2)

def sample(args):
    hole_card, round_state, depth = args
    state = State(hole_card, round_state)
    _, best_move = minimax(state, depth, is_max=True)
    return best_move

class TestPlayer(BasePokerPlayer):       
    def declare_action(self, valid_actions, hole_card, round_state):    
        samples = 10
        depth = 3
        work = [(hole_card, round_state, depth) for _ in range(samples)]
        with ThreadPoolExecutor() as pool:
            moves = list(pool.map(sample, work))
        move_votes = Counter(moves)
        return move_votes.most_common(1)[0][0]

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def setup_ai():
    return TestPlayer()

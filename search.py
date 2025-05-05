import pprint

from math import inf
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.round_manager import RoundManager
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate
from pypokerengine.engine.action_checker import ActionChecker

pp = pprint.PrettyPrinter(indent=2)

def minimax(game_state, depth, is_max):
    if (game_state["street"] == Const.Street.FINISHED) or depth == 0:
        # TODO: call evaluation function
        return 1, None

    # Generate legal actions at current state
    actions = ActionChecker.legal_actions(
        game_state["table"].seats.players, 
        game_state["next_player"],
        game_state["small_blind_amount"],
        game_state["street"]
    )

    top_score = -inf if is_max else inf
    top_action = None

    print(actions)

    # Search actions recursively
    for action in actions:
        next_state, _ = RoundManager.apply_action(game_state, action["action"])   
        score, _ = minimax(next_state, depth - 1, not is_max)

        # Update if new best found
        if (is_max and score > top_score) or (not is_max and score < top_score):
            top_score, top_action = score, action

    return top_score, top_action

from math import inf
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate
from pypokerengine.engine.action_checker import ActionChecker

import pprint

def evaluate_leaf(table, uuid):
    player = next(player for player in table.seats.players if player.uuid == uuid)
    return estimate_hole_card_win_rate(
        nb_simulation=100,
        nb_player=2,
        hole_card=player.hole_card,
        community_card=table.community_card
    )

def minimax(emulator, game_state, uuid, depth, is_max):
    terminal = (game_state["street"] == Const.Street.FINISHED) or depth == 0
    if terminal:
        leaf_score = evaluate_leaf(game_state["table"], uuid)
        return leaf_score, None

    best_score = -inf if is_max else inf
    best_action = None

    # Generate legal actions at current state
    actions = ActionChecker.legal_actions(
        game_state["table"].seats.players, 
        game_state["next_player"],
        game_state["small_blind_amount"],
        game_state["street"]
    )

    for action in actions:
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(action)

        # Generate state from selected action
        next_state, _ = emulator.apply_action(game_state, action["action"])   

        # Recur on next state
        score, _ = minimax(emulator, next_state, uuid, depth - 1, not is_max)

        if is_max and score > best_score:
            best_score, best_action = score, action
        if not is_max and score < best_score:
            best_score, best_action = score, action

    return best_score, best_action

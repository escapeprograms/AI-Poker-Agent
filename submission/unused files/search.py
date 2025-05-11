import pprint

from math import inf
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.round_manager import RoundManager
from pypokerengine.engine.action_checker import ActionChecker
from encode_state import encode_game_state

pp = pprint.PrettyPrinter(indent=2)

def minimax(game_state, events, depth, is_max, value_network):
    print(game_state)
    if (game_state["street"] == Const.Street.FINISHED):
        print("FINISHED")
        winners = events[-1][1]["message"]["winners"]
        for winner in winners:
            print(is_max)
            if winner["uuid"] == game_state["table"].seats.players[1-game_state["next_player"]].uuid:
                return 60, None
        return -60, None
        #TODO: replace all 60 with the pot of the round
    if depth == 0:
        print("DEPTH 0")
        # events[-1][1]["message"]["winners"] stores winner
        round_state = events[-1][1]["message"]["round_state"]
        # print("round state:", round_state)
        hole_card = game_state['table'].seats.players[0 if is_max else 1].hole_card
        val = value_network(*encode_game_state(hole_card, round_state))
        if len(hole_card) > 0:
            print(str(hole_card[0]), str(hole_card[1]))
        print(val)
        return val, None
    

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
        score, _ = minimax(next_state, events, depth - 1, not is_max, value_network)
        if (is_max and score > top_score) or (not is_max and score < top_score):
            top_score, top_action = score, action
        print(score,action)
    return top_score, top_action




# Change timeout2 call in pypokerengine/api/game to use this
# def manual_walk(game_state, event=None):
    # print("\nGame state:")
    # print(f"   Player:    {game_state['next_player']}")
    # print(f"   Street:    {game_state['street']}")
    # print(f"   Community: {[str(card) for card in game_state['table']._community_card]}")
    # print(f"   Player:    {game_state['next_player']}")
    # print(f"   Hole:      {[str(card) for card in game_state['table'].seats.players[game_state['next_player']].hole_card]}")

    # if (game_state["street"] == Const.Street.FINISHED):
    #     # print(f"\nWinner: {event[1]["message"]["winners"]}")
    #     quit()

    # actions = ActionChecker.legal_actions(
    #     game_state["table"].seats.players, 
    #     game_state["next_player"],
    #     game_state["small_blind_amount"],
    #     game_state["street"]
    # )

    # print(f"\nAvailable actions:")
    # for i, entry in enumerate(actions, start=1):
    #     # print(f"   {i}. {entry['action']}")
    # choice = input("> ")
    # next_state, events = RoundManager.apply_action(game_state, actions[int(choice) - 1]["action"])   
    # manual_walk(next_state, events[-1])
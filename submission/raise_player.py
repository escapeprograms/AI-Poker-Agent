from pypokerengine.players import BasePokerPlayer
from time import sleep
import pprint

class RaisedPlayer(BasePokerPlayer):

  def declare_action(self, valid_actions, hole_card, round_state):
    # print("hole_card",hole_card)
    # print("round_state", round_state)
    # print(round_state['street'])
    for i in valid_actions:
        if i["action"] == "raise":
            action = i["action"]
            return action  # action returned here is sent to the poker engine
    action = valid_actions[1]["action"]
    return action # action returned here is sent to the poker engine

  def receive_game_start_message(self, game_info):
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    pass

  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass

def setup_ai():
  return RaisedPlayer()


class CallPlayer(BasePokerPlayer):

  def declare_action(self, valid_actions, hole_card, round_state):
    # print("hole_card",hole_card)
    # print("round_state", round_state)
    # print(round_state['street'])
    # for i in valid_actions:
    #     if i["action"] == "raise" and (round_state['street'] == 'flop' or round_state['street'] == 'river'or round_state['street'] == 'flop'):
    #         action = i["action"]
    #         return action  # action returned here is sent to the poker engine
    action = valid_actions[1]["action"]
    return action # action returned here is sent to the poker engine

  def receive_game_start_message(self, game_info):
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    pass

  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass

class FoldPlayer(BasePokerPlayer):
  def declare_action(self, valid_actions, hole_card, round_state):
    # print("hole_card",hole_card)
    # print("round_state", round_state)
    # print(round_state['street'])
    # for i in valid_actions:
    #     if i["action"] == "raise" and (round_state['street'] == 'flop' or round_state['street'] == 'river'or round_state['street'] == 'flop'):
    #         action = i["action"]
    #         return action  # action returned here is sent to the poker engine
    action = valid_actions[0]["action"]
    return action # action returned here is sent to the poker engine

  def receive_game_start_message(self, game_info):
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    pass

  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass
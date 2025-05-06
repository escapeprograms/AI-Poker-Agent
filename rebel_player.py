from pypokerengine.players import BasePokerPlayer
from time import sleep
import pprint
from tree import Node

class RebelPlayer(BasePokerPlayer):
    def __init__(self, value_network):
        super().__init__()
        self.value_network = value_network

    def encode_state(self, hole_card, round_state):
        #find the belief state
        #encode the round state into a vector for the value function
        
        return []
    
    def build_tree(self, policy, hole_card, round_state):
        # policy = [0.33,0.33,0.34]
        root = Node(0)
        for i in ["fold","call","raise"]:
            #update the round state

            root.children.append(self.build_tree(policy, hole_card, round_state))
        return root
    
    def declare_action(self, valid_actions, hole_card, round_state):
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
  return RebelPlayer()

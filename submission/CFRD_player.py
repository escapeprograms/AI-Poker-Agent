import random
from pypokerengine.players import BasePokerPlayer
from encode_state import encode_game_state
from pypokerengine.utils.game_state_utils import restore_game_state

class CFRDPlayer(BasePokerPlayer):       
    def __init__(self, value_network, device='cuda'):
        super().__init__()
        self.value_network = value_network
        self.device = device

    # Static method to get the policy
    def get_policy(game_state, hole_card, actions, value_network, cur_player, eval_device="cpu"):
        pred_vals = []
        policy = []
        for action in actions:
            # Calculate the value of each action
            val = value_network(*encode_game_state(hole_card, game_state, action, cur_player, device=eval_device)).item()
            pred_vals.append(val) # Don't allow negative values

        # Calculate the policy
        total_val = sum([max(0, val) for val in pred_vals])

        # If all values are negative, use the max value
        if total_val == 0:
            policy = [0, 0, 0]
            policy[pred_vals.index(max(pred_vals))] = 1
        else:
            # Define policy based on values
            for val in pred_vals:
                probability = max(val, 0) / total_val
                policy.append(probability)
        return pred_vals, policy

    def declare_action(self, valid_actions, hole_card, round_state):

        seat = round_state["next_player"]            # Get agent's UUID
        game_state = restore_game_state(round_state) # Get the game state   

        # Copied from train_value_model.py simulation
        pred_vals, policy = CFRDPlayer.get_policy(game_state, hole_card, valid_actions, self.value_network, seat, eval_device=self.device)
        
        # Choose an action to take
        action_indices = list(range(len(policy)))
        selected_action = random.choices(action_indices, weights=policy, k=1)[0]
        return valid_actions[selected_action]["action"]

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def setup_ai():
    return CFRDPlayer()

import pprint
import random

from pypokerengine.players import BasePokerPlayer
from encode_state import encode_game_state

from math import inf

from pypokerengine.utils.game_state_utils import restore_game_state

pp = pprint.PrettyPrinter(indent=2)

class CFRDPlayer(BasePokerPlayer):       
    def __init__(self, value_network, device='cuda', verbose_print = False):
        super().__init__()
        self.value_network = value_network
        self.device = device
        self.verbose_print = verbose_print

    #static method to get the policy
    def get_policy(game_state, hole_card, actions, value_network, cur_player, eval_device="cpu"):
        pred_regrets = []
        pred_vals = []
        policy = []
        for action in actions:
            if action["action"] == "fold": #get direct value for fold
                bet = 0
                past_actions = game_state["table"].seats.players[cur_player].round_action_histories

                #add the blind amount to the bet
                if past_actions[0] is not None and past_actions[0][0]['action'] == "BIGBLIND":
                    bet = game_state["small_blind_amount"] * 2
                else:
                    bet = game_state["small_blind_amount"]
                #check all bets
                for i in range(len(past_actions)):
                    street_actions = past_actions[i]
                    if street_actions == None:
                        break
                    for j in range(len(street_actions)):
                        if street_actions[j]['action'] in ["FOLD","SMALLBLIND","BIGBLIND"]:
                            continue
                        bet += street_actions[j]['paid'] 
                
                pred_regrets.append(-bet)
                pred_vals.append(-bet)
                continue
            
            #calculate the value of each other action
            regret, val = value_network(*encode_game_state(hole_card, game_state, action, cur_player, device=eval_device)).tolist()
            pred_regrets.append(regret)
            pred_vals.append(val)

        #calculate the policy
        total_regret = sum([max(0, regret) for regret in pred_regrets])

        #if all regrets are negative, use the max value
        if total_regret == 0:
            policy = [0, 0, 0]
            policy[pred_regrets.index(max(pred_regrets))] = 1
        else:
            #define policy based on values
            for regret in pred_regrets:
                probability = max(regret, 0) / total_regret
                policy.append(probability)
        return pred_vals, policy #return values and policy for each action 

    def declare_action(self, valid_actions, hole_card, round_state):


        # Get agent's UUID
        seat = round_state["next_player"]
        uuid = round_state["seats"][seat]["uuid"]
        # Get the game state
        game_state = restore_game_state(round_state)        

        #Get hole card
        if self.verbose_print and len(hole_card) > 0:
            print("Hole card", hole_card)

        #copied from train_value_model.py simulation
        pred_vals, policy = CFRDPlayer.get_policy(game_state, hole_card, valid_actions, self.value_network, seat, eval_device=self.device)
        
        if self.verbose_print:
            print("Pred vals", pred_vals)
            print("Policy", policy)
        #choose an action to take
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

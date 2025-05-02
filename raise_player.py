from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import pprint

class RaisedPlayer(BasePokerPlayer):
    def __init__(self):
        self.win_rate_threshold_to_raise = 0.6
        self.win_rate_threshold_to_call = 0.3
        self.num_simulations = 100
        self.opponent_fold_freq = 0.0
        self.opponent_history = []

    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=self.num_simulations,
            nb_player=len(round_state['seats']),
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_card)
        )
        street = round_state['street']
        pot = round_state['pot']['main']['amount']
        current_bet = max([action["amount"] for action in round_state["action_histories"].get(street, [])], default=0)
        pot_odds = current_bet / (pot + current_bet) if pot + current_bet > 0 else 0

        # Update opponent fold frequency
        self.opponent_history.extend([a["action"] for a in round_state["action_histories"].get(street, []) if a["action"] == "fold"])
        self.opponent_fold_freq = len(self.opponent_history) / max(len(round_state['seats']) - 1, 1) if len(round_state['seats']) > 1 else 0.0

        my_pos = -1
        for i, player in enumerate(round_state['seats']):
            if player['uuid'] == self.uuid:
                my_pos = i
        total_players = len(round_state['seats'])
        is_late_position = my_pos >= total_players - 2

        # Adjust thresholds based on street and position
        if street == 'flop':
            self.win_rate_threshold_to_raise = 0.65 - (0.05 if is_late_position else 0)
        elif street == 'turn':
            self.win_rate_threshold_to_raise = 0.7 - (0.05 if is_late_position else 0)
        elif street == 'river':
            self.win_rate_threshold_to_raise = 0.75 - (0.05 if is_late_position else 0)

        can_raise = len(valid_actions) >= 3

        if win_rate >= self.win_rate_threshold_to_raise + (0.1 * self.opponent_fold_freq) and can_raise and pot_odds < win_rate:
            action = valid_actions[2]["action"]
            if isinstance(valid_actions[2].get("amount"), dict):
                min_raise = valid_actions[2]["amount"]["min"]
                max_raise = valid_actions[2]["amount"]["max"]
                raise_factor = 0.3 + 0.2 * (pot / 10000)  # Adjust based on pot size
                raise_amount = min(min_raise + (max_raise - min_raise) * raise_factor, max_raise)
                return action, int(raise_amount)
            return action
        elif win_rate >= self.win_rate_threshold_to_call and pot_odds < win_rate:
            action = valid_actions[1]["action"]
            return action
        else:
            action = valid_actions[0]["action"]
            return action

    def receive_game_start_message(self, game_info):
        self.num_players = len(game_info['seats'])
        self.opponent_history = []

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        for winner in winners:
            if winner['uuid'] == self.uuid:
                self.win_rate_threshold_to_call = max(0.25, self.win_rate_threshold_to_call - 0.02)
                self.win_rate_threshold_to_raise = max(0.55, self.win_rate_threshold_to_raise - 0.02)
            else:
                self.win_rate_threshold_to_call = min(0.4, self.win_rate_threshold_to_call + 0.02)
                self.win_rate_threshold_to_raise = min(0.7, self.win_rate_threshold_to_raise + 0.02)

def setup_ai():
    return RaisedPlayer()
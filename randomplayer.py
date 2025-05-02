from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import random as rand
import pprint

class RandomPlayer(BasePokerPlayer):
    def __init__(self):
        self.fold_threshold = 0.3
        self.raise_threshold = 0.7
        self.bluff_frequency = 0.1
        self.win_rate_threshold = 0.4
        self.num_simulations = 100
        self.total_rounds = 0
        self.hands_played = 0
        self.opponent_raise_freq = 0.0
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

        # Update opponent raise frequency
        self.opponent_history.extend([a["action"] for a in round_state["action_histories"].get(street, []) if a["action"] == "raise"])
        self.opponent_raise_freq = len(self.opponent_history) / max(self.total_rounds, 1) if self.total_rounds > 0 else 0.0

        my_pos = -1
        for i, player in enumerate(round_state['seats']):
            if player['uuid'] == self.uuid:
                my_pos = i
        total_players = len(round_state['seats'])
        is_late_position = my_pos >= total_players - 2

        # Adjust thresholds
        if street == 'preflop':
            self.fold_threshold = 0.25
            self.raise_threshold = 0.75
        elif street == 'flop':
            self.fold_threshold = 0.3
            self.raise_threshold = 0.7
        elif street == 'turn':
            self.fold_threshold = 0.35
            self.raise_threshold = 0.65
        else:
            self.fold_threshold = 0.4
            self.raise_threshold = 0.6

        # Adjust bluff frequency based on opponent
        adjusted_bluff_freq = self.bluff_frequency * (1 - self.opponent_raise_freq)

        bluffing = rand.random() < adjusted_bluff_freq

        # Decision making logic with pot odds
        if (win_rate >= self.raise_threshold or (is_late_position and bluffing)) and pot_odds < win_rate:
            if len(valid_actions) >= 3:
                action = valid_actions[2]["action"]
                if isinstance(valid_actions[2].get("amount"), dict):
                    min_raise = valid_actions[2]["amount"]["min"]
                    max_raise = valid_actions[2]["amount"]["max"]
                    raise_amount = min_raise + (max_raise - min_raise) * 0.3
                    return action, int(max(min_raise, raise_amount))
                return action
            action = valid_actions[1]["action"]
            return action
        elif win_rate >= self.fold_threshold and pot_odds < win_rate:
            action = valid_actions[1]["action"]
            return action
        else:
            action = valid_actions[0]["action"]
            return action

    def receive_game_start_message(self, game_info):
        self.total_rounds = 0
        self.hands_played = 0
        self.opponent_history = []

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.total_rounds += 1
        self.hands_played += 1

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        for winner in winners:
            if winner['uuid'] == self.uuid:
                self.fold_threshold = max(0.2, self.fold_threshold - 0.05)
                self.raise_threshold = max(0.6, self.raise_threshold - 0.05)

def setup_ai():
    return RandomPlayer()
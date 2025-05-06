import pprint

from pypokerengine.players import BasePokerPlayer
from search import minimax, manual_walk
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.utils.game_state_utils import attach_hole_card, attach_hole_card_from_deck, restore_game_state

pp = pprint.PrettyPrinter(indent=2)

class MinimaxPlayer(BasePokerPlayer):       
    def __init__(self, value_network, train=False):
        super().__init__()
        self.value_network = value_network

         #after each round, store some training data from the round
        self.train_hole_cards = []
        self.train_round_states = []
        self.train_values = []
        
        self.cur_round_states = [] #keep previous round states to use for training
        self.hole_card = [] #save hole card

    def declare_action(self, valid_actions, hole_card, round_state):
        self.cur_round_states.append(round_state) #add round state to training list
        self.hole_card = hole_card #save hole card

        # Get agent's UUID
        seat = round_state["next_player"]
        uuid = round_state["seats"][seat]["uuid"]

        # Clone known game state
        game_state = restore_game_state(round_state)            
        game_state = attach_hole_card(game_state, uuid, gen_cards(hole_card))
        # print("gamestate", game_state)
        # Remove hole cards from deck
        for card in gen_cards(hole_card):
            game_state["table"].deck.deck.remove(card)

        # Generate random hole cards for opponents
        for seat in round_state["seats"]:
            if seat["uuid"] != uuid:
                game_state = attach_hole_card_from_deck(game_state, seat["uuid"])

        # Search for best action
        return minimax(game_state, None, 2, is_max=True, value_network=self.value_network)[1]["action"]

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state):
        #after each round, store training data containing the hole cards, a partial state, and the value (money won/lost)
        value = round_state['pot']['main']['amount'] * (1 if winners[0]['uuid'] == self.uuid else -1)
        # Store multiple training data
        for state in self.cur_round_states:
            keys_to_keep = ["community_card", "action_histories"]
            subset_state = {key: state[key] for key in keys_to_keep if key in state}
            
            self.train_hole_cards.append(self.hole_card)
            self.train_round_states.append(subset_state)
            self.train_values.append(value)
            
        self.cur_round_states = [] #reset stored round states
        self.hole_card = []

def setup_ai():
    return MinimaxPlayer()

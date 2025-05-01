import random
from dataclasses import dataclass
from copy import deepcopy
from typing import List, Tuple
from math import inf

from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards

MAX_RAISES = 4
RANKS = '23456789TJQKA'
SUITS = 'CDHS'
FULL_DECK = [f"{suit}{rank}" for suit in SUITS for rank in RANKS]

class State:
    def __init__(self, hole_card, round_state):
        # Easily extractable state
        self.maxer = round_state["next_player"]
        self.player = round_state["next_player"]
        self.community_card = round_state["community_card"]
        self.pot = round_state["pot"]["main"]["amount"]
        self.stages = ['preflop','flop','turn','river']
        self.stage = self.stages.index(round_state["street"])
        self.raise_amount = round_state["small_blind_amount"] * (2 if self.stage < 2 else 4)
        self.stacks = [
            round_state["seats"][0]["stack"],
            round_state["seats"][1]["stack"]
        ] 
        self.terminal = False
        self.winner = None

        # Setup holes
        self.holes = [None, None]
        self.holes[self.player] = [card.__str__() for card in hole_card]

        # Sample random cards for opponent
        used = set(self.community_card) | set(self.holes[self.player])
        deck = [card for card in FULL_DECK if card not in used]
        self.holes[1 - self.player] = random.sample(deck, 2)
        
        # Extract raises from history
        self.raises = [0, 0] 
        uuids = [None, None]
        uuids[self.player] = round_state["seats"][self.player]["uuid"]
        uuids[1 - self.player] = round_state["seats"][1 - self.player]["uuid"]
        for action in round_state["action_histories"][round_state["street"]]:
            if action["action"] == "RAISE":
                self.raises[uuids.index(action["uuid"])] += 1

        # Extract bet to call
        max_bets = {}
        max_bets[uuids[0]] = 0
        max_bets[uuids[1]] = 0
        for action in round_state["action_histories"][round_state["street"]]:
            max_bets[action["uuid"]] = max(max_bets[action["uuid"]], action["amount"])
        self.to_call = max_bets[uuids[1 - self.player]] - max_bets[uuids[self.player]] 

    def __str__(self):
        return (
            f"State(\n" 
            f"  player:    {self.player}\n"
            f"  stage:     {self.stage} ({self.stages[self.stage]})\n"
            f"  pot:       {self.pot}\n"
            f"  to_call:   {self.to_call}\n"
            f"  stacks:    {self.stacks}\n"
            f"  raises:    {self.raises}\n"
            f"  holes:     {self.holes}\n"
            f"  community: {self.community_card}\n"
            f"  terminal:  {self.terminal}\n"
            f")"
        )

def legal_actions(state: State) -> List[str]:
    actions = ['fold']
    if state.stacks[state.player] >= state.to_call: # TODO: Handle "all-in" case?
        actions.append('call')
    if state.raises[state.player] < MAX_RAISES and state.stacks[state.player] >= state.to_call + state.raise_amount:
        actions.append('raise')
    return actions

def apply_action(state: State, action: str) -> State:
    # Create new states
    state = deepcopy(state)

    if action == 'fold':
        # Other player wins pot and game ends
        state.stacks[1 - state.player] += state.pot
        state.pot = 0
        state.terminal = True
        return state
    
    elif action == 'raise':
        # Raise bet by small blind
        state.raises[state.player] += 1
        amount = state.to_call + state.raise_amount
        state.stacks[state.player] -= amount
        state.pot += amount
        state.to_call = state.raise_amount

    elif action == 'call':
        # Level bets
        state.stacks[state.player] -= state.to_call
        state.pot += state.to_call
        state.to_call = 0

        # Advance stage since bets are even
        if state.stage < len(state.stages) - 1:
            state.stage += 1
            if state.stage == 2:
                state.raise_amount *= 2 # Raise amount doubles after flop

            # Deal cards
            used = set(state.community_card) | set(state.holes[0]) | set(state.holes[1])
            deck = [card for card in FULL_DECK if card not in used]
            count = 3 if state.stage == 1 else 1
            state.community_card.extend(random.sample(deck, count))
            state.raises = [0, 0] 
        else:
            community = gen_cards(state.community_card)
            if HandEvaluator.eval_hand(gen_cards(state.holes[state.player]), community) \
            >= HandEvaluator.eval_hand(gen_cards(state.holes[1 - state.player]), community):
                state.stacks[state.player] += state.pot
                state.winner = state.player
            else:
                state.stacks[1 - state.player] += state.pot
                state.winner = 1 - state.player
            state.terminal = True
            return state

    # Swap players and continue
    state.player = 1 - state.player
    return state

def evaluate(state: State) -> float:
    maxer_win_rate = estimate_hole_card_win_rate(
        nb_simulation = 100,
        nb_player = 2,
        hole_card = gen_cards(state.holes[state.maxer]),
        community_card = gen_cards(state.community_card)
    )

    # TODO: fix this, it's not right
    return maxer_win_rate * state.pot

def minimax(state: State, depth: int, is_max: bool) -> Tuple[float, str]:
    if state.terminal:
        return state.stacks[state.maxer] - state.stacks[1 - state.maxer], None
    
    if depth == 0:
        return evaluate(state), None

    best_score = -inf if is_max else inf
    best_action = None

    for action in legal_actions(state):
        next_state = apply_action(state, action)
        score, _  = minimax(next_state, depth - 1, not is_max)

        if is_max:
            if score > best_score:
                best_score, best_action = score, action
        else:
            if score < best_score:
                best_score, best_action = score, action
    return best_score, best_action
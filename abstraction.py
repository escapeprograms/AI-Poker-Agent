# Poker agent with enhanced abstraction for COMPSCI 683 project
# Optimized for efficiency, training support, and opponent modeling

import random
import time
from typing import List, Tuple, Optional

# Global variables for training and opponent modeling
global_weights = {
    "card_weight": 0.7,       # Weight for card bucket
    "bet_weight": 0.3,        # Weight for betting bucket
    "threshold_premium": 0.85, # Threshold for premium hands
    "threshold_strong": 0.7,   # Threshold for strong hands
    "bluff_prob": 0.2         # Probability of bluffing
}
opponent_history = {"raises": 0, "calls": 0, "folds": 0}  # Track opponent actions

def get_hand_strength(hole_cards: List[str], community_cards: List[str]) -> float:
    """
    Enhanced hand strength evaluation (0 to 1).
    Evaluates card ranks, made hands, and potential hands (flush, straight).
    Handles pre-flop and post-flop scenarios.
    """
    if not community_cards:  # Pre-flop evaluation
        return evaluate_preflop(hole_cards)
    
    # Card values for evaluation
    ranks = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
             'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    
    # Count pairs, three/four of a kind
    rank_count = {}
    for card in hole_cards + community_cards:
        rank = card[0]
        rank_count[rank] = rank_count.get(rank, 0) + 1
    
    # Check for pairs and better
    pairs = sum(1 for count in rank_count.values() if count >= 2)
    three_kind = any(count >= 3 for count in rank_count.values())
    four_kind = any(count >= 4 for count in rank_count.values())
    
    # Check for flush potential
    suits = {}
    for card in hole_cards + community_cards:
        suit = card[1]
        suits[suit] = suits.get(suit, 0) + 1
    
    flush_potential = max(suits.values() if suits else 0)
    
    # Basic strength evaluation
    strength = 0.0
    if four_kind:
        strength = 0.9  # Four of a kind
    elif three_kind and pairs >= 2:
        strength = 0.8  # Full house
    elif flush_potential >= 5:
        strength = 0.75  # Flush
    elif three_kind:
        strength = 0.7  # Three of a kind
    elif pairs >= 2:
        strength = 0.6  # Two pair
    elif pairs == 1:
        strength = 0.5  # One pair
    else:
        # High card - normalize based on highest cards
        hole_values = [ranks[card[0]] for card in hole_cards]
        community_values = [ranks[card[0]] for card in community_cards]
        all_values = sorted(hole_values + community_values, reverse=True)
        top_five = all_values[:5] if len(all_values) >= 5 else all_values
        strength = sum(top_five) / 70.0  # Normalize to [0, 1]
    
    return min(strength, 1.0)

def evaluate_preflop(hole_cards: List[str]) -> float:
    """
    Evaluate pre-flop hand strength.
    Higher value for premium starting hands (pocket pairs, suited connectors).
    """
    ranks = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
             'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    
    # Extract rank and suit
    rank1, suit1 = hole_cards[0][0], hole_cards[0][1]
    rank2, suit2 = hole_cards[1][0], hole_cards[1][1]
    value1, value2 = ranks[rank1], ranks[rank2]
    
    # Sort values for easier comparison
    high, low = max(value1, value2), min(value1, value2)
    
    # Pocket pairs are strong
    if rank1 == rank2:
        return 0.5 + (high / 28.0)  # Normalize to [0.5, 1.0]
    
    # Suited cards
    suited = suit1 == suit2
    
    # Premium hands (high cards, suited connectors)
    if high >= 12 and low >= 10:  # High cards (A, K, Q, J, T)
        return 0.6 + (suited * 0.1)
    elif high - low <= 2 and suited:  # Suited connectors
        return 0.5 + (high / 28.0)
    elif high >= 13:  # At least one Ace or King
        return 0.4 + (low / 28.0) + (suited * 0.1)
    
    # Lower value hands
    return 0.2 + (high / 28.0) + (low / 56.0) + (suited * 0.1)

def bucket_cards(hole_cards: List[str], community_cards: List[str]) -> str:
    """
    Enhanced card abstraction with strategic categories.
    Buckets: premium_hand, strong_hand, medium_hand, drawing_hand, weak_hand.
    Considers made hands and drawing potential (flush, straight).
    """
    # Get hand strength
    strength = get_hand_strength(hole_cards, community_cards)
    
    # Count suits for flush potential
    suits = {}
    for card in hole_cards + community_cards:
        suit = card[1]
        suits[suit] = suits.get(suit, 0) + 1
    
    # Count ranks for pairs or higher
    ranks = {}
    for card in hole_cards + community_cards:
        rank = card[0]
        ranks[rank] = ranks.get(rank, 0) + 1
    
    # Check for flush potential (4+ cards of same suit)
    flush_potential = any(count >= 4 for count in suits.values())
    
    # Check for straight potential (connected cards)
    card_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                  'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    all_values = sorted([card_values[card[0]] for card in hole_cards + community_cards])
    straight_potential = False
    
    if len(all_values) >= 4:  # Need at least 4 cards for straight potential
        consecutive_count = 1
        for i in range(1, len(all_values)):
            if all_values[i] == all_values[i-1] + 1:
                consecutive_count += 1
                if consecutive_count >= 4:  # 4 consecutive for straight potential
                    straight_potential = True
                    break
            elif all_values[i] != all_values[i-1]:
                consecutive_count = 1
    
    # Define drawing hand: strong draw to straight or flush
    drawing_hand = flush_potential or straight_potential
    
    # Check for made hands
    has_pair = any(count >= 2 for count in ranks.values())
    has_trips = any(count >= 3 for count in ranks.values())
    has_quads = any(count >= 4 for count in ranks.values())
    
    # Advanced bucketing based on strength and potential
    if strength > 0.8 or has_quads:
        return "premium_hand"  # Very strong made hands
    elif strength > 0.6 or has_trips:
        return "strong_hand"   # Strong made hands
    elif strength > 0.5 or has_pair:
        return "medium_hand"   # Medium strength hands and strong pair
    elif drawing_hand:
        return "drawing_hand"  # Drawing hands with potential
    else:
        return "weak_hand"     # Weak hands with little potential

def bucket_betting(pot_size: float, my_bet: float, opp_bet: float) -> str:
    """
    Enhanced betting abstraction considering pot odds and commitment.
    Buckets: small_pot, medium_pot, large_pot, huge_pot.
    Integrates pot odds to reflect strategic favorability.
    """
    # Calculate total pot size including current bets
    total_pot = pot_size + my_bet + opp_bet
    
    # Calculate pot odds (ratio of what you can win vs what you must call)
    if opp_bet > my_bet:
        pot_odds = total_pot / (opp_bet - my_bet)
    else:
        pot_odds = float('inf')
    
    # Pot commitment ratio (how much already invested)
    commitment_ratio = my_bet / total_pot if total_pot > 0 else 0
    
    # Nuanced pot size buckets with pot odds consideration
    if total_pot <= 30 or pot_odds > 5:
        return "small_pot"  # Favorable odds or small pot
    elif total_pot <= 60:
        return "medium_pot"
    elif total_pot <= 100:
        return "large_pot"
    else:
        return "huge_pot"

def model_opponent(opp_bet: float, pot_size: float, my_bet: float, card_bucket: str) -> str:
    """
    Enhanced opponent modeling based on current and historical betting patterns.
    Returns opponent type: "passive", "aggressive", "balanced".
    Uses opponent_history for more accurate classification.
    """
    # Update opponent history (assumes template provides opponent action)
    # Note: Actual update requires game loop feedback, simplified here
    total_actions = sum(opponent_history.values()) + 1
    raise_freq = opponent_history["raises"] / total_actions if total_actions > 0 else 0
    
    # Classify based on current bet and historical frequency
    if raise_freq > 0.5 or opp_bet > pot_size * 0.5:
        return "aggressive"
    elif raise_freq < 0.2 or (opp_bet == 0 or opp_bet == my_bet):
        return "passive"
    else:
        return "balanced"

def evaluate_state(card_bucket: str, bet_bucket: str, opp_behavior: str, pot_odds: float) -> float:
    """
    Enhanced evaluation function combining abstracted states and pot odds.
    Uses nuanced weights and strategic adjustments for card strength, pot size, and opponent behavior.
    """
    # Nuanced weights for card buckets
    card_weights = {
        "premium_hand": 0.95,
        "strong_hand": 0.8,
        "medium_hand": 0.6,
        "drawing_hand": 0.5,
        "weak_hand": 0.2
    }
    
    # Weights for betting buckets
    bet_weights = {
        "small_pot": 0.4,
        "medium_pot": 0.6,
        "large_pot": 0.7,
        "huge_pot": 0.8
    }
    
    # Opponent behavior modifiers
    behavior_modifiers = {
        "passive": 1.2,   # Exploit passive players
        "balanced": 1.0,  # Solid play against balanced
        "aggressive": 0.8 # Cautious against aggressive
    }
    
    # Strategic adjustments based on card strength, pot size, and pot odds
    strategy_adjustment = 0.0
    
    # Premium hands in big pots
    if card_bucket == "premium_hand" and bet_bucket in ["large_pot", "huge_pot"]:
        strategy_adjustment = 0.1
    # Weak hands in big pots with poor odds
    elif card_bucket == "weak_hand" and bet_bucket in ["large_pot", "huge_pot"] and pot_odds < 2:
        strategy_adjustment = -0.1
    # Drawing hands with favorable odds
    elif card_bucket == "drawing_hand" and bet_bucket in ["small_pot", "medium_pot"] and pot_odds > 3:
        strategy_adjustment = 0.05
    
    # Combined evaluation with weighted components
    score = (global_weights["card_weight"] * card_weights[card_bucket] + 
             global_weights["bet_weight"] * bet_weights[bet_bucket]) * behavior_modifiers[opp_behavior]
    
    # Apply strategic adjustment
    score += strategy_adjustment
    
    # Ensure score in [0, 1]
    return max(0.0, min(1.0, score))

def simulate_action(state: dict, action: Tuple[str, Optional[float]]) -> dict:
    """
    Simulate the outcome of an action for minimax search.
    Simplified simulation assuming opponent responds conservatively.
    Returns updated state for next search level.
    """
    new_state = state.copy()
    action_type, amount = action
    
    if action_type == "raise":
        new_state["my_bet"] = amount
        new_state["pot_size"] += (amount - state["my_bet"])
        # Assume opponent calls or folds (simplified)
        if random.random() < 0.5:
            new_state["opp_bet"] = amount
            new_state["pot_size"] += (amount - state["opp_bet"])
        else:
            new_state["game_over"] = True  # Opponent folds
    elif action_type == "call":
        new_state["my_bet"] = state["opp_bet"]
        new_state["pot_size"] += (state["opp_bet"] - state["my_bet"])
    elif action_type == "fold":
        new_state["game_over"] = True
    elif action_type == "check":
        pass  # No change in bets
    
    return new_state

def minimax(state: dict, depth: int, is_max: bool) -> float:
    """
    Simplified minimax search for 1-2 layers.
    Evaluates actions using evaluate_state at cutoff.
    Assumes conservative opponent for simulation.
    """
    if depth == 0 or state.get("game_over", False):
        return evaluate_state(state["card_bucket"], state["bet_bucket"], 
                            state["opp_behavior"], state["pot_odds"])
    
    actions = state["valid_actions"]
    if is_max:  # AI's turn
        best = float('-inf')
        for action in actions:
            next_state = simulate_action(state, action)
            score = minimax(next_state, depth - 1, False)
            best = max(best, score)
        return best
    else:  # Opponent's turn
        best = float('inf')
        for action in actions:  # Simplified: assume similar actions
            next_state = simulate_action(state, action)
            score = minimax(next_state, depth - 1, True)
            best = min(best, score)
        return best

def update_opponent_history(opp_action: str):
    """
    Update opponent action history for modeling.
    Called by game loop (assumed provided by template).
    """
    if opp_action == "raise":
        opponent_history["raises"] += 1
    elif opp_action == "call":
        opponent_history["calls"] += 1
    elif opp_action == "fold":
        opponent_history["folds"] += 1

def declare_action(hole_cards: List[str], community_cards: List[str], 
                  my_bet: float, opp_bet: float, pot_size: float, 
                  valid_actions: List[Tuple[str, Optional[float]]]) -> Tuple[str, Optional[float]]:
    """
    Enhanced decision-making function using abstracted states and minimax search.
    Integrates card and betting abstraction, opponent modeling, and pot odds.
    Logs decision time for performance analysis.
    """
    # Log start time for performance
    start_time = time.time()
    
    # Step 1: Abstract the card state
    card_bucket = bucket_cards(hole_cards, community_cards)
    
    # Step 2: Abstract the betting state
    bet_bucket = bucket_betting(pot_size, my_bet, opp_bet)
    
    # Step 3: Model opponent behavior
    opp_behavior = model_opponent(opp_bet, pot_size, my_bet, card_bucket)
    
    # Step 4: Calculate pot odds
    total_pot = pot_size + my_bet + opp_bet
    pot_odds = total_pot / (opp_bet - my_bet) if opp_bet > my_bet else float('inf')
    
    # Step 5: Prepare state for minimax
    state = {
        "card_bucket": card_bucket,
        "bet_bucket": bet_bucket,
        "opp_behavior": opp_behavior,
        "pot_odds": pot_odds,
        "my_bet": my_bet,
        "opp_bet": opp_bet,
        "pot_size": pot_size,
        "valid_actions": valid_actions
    }
    
    # Step 6: Use minimax search (depth 2) to evaluate actions
    scores = []
    for action in valid_actions:
        next_state = simulate_action(state, action)
        score = minimax(next_state, 2, False)
        scores.append((action, score))
    
    # Step 7: Select best action with randomization for bluffs
    best_action, best_score = max(scores, key=lambda x: x[1])
    to_call = opp_bet - my_bet
    
    # Bluffing logic for weak/drawing hands
    if card_bucket in ["weak_hand", "drawing_hand"] and bet_bucket in ["small_pot", "medium_pot"]:
        if random.random() < global_weights["bluff_prob"] and ("raise", my_bet + 10) in valid_actions:
            best_action = ("raise", my_bet + 10)
    
    # Adjust for aggressive opponents
    if opp_behavior == "aggressive" and best_score < global_weights["threshold_strong"]:
        if random.random() < 0.3 and ("call", None) in valid_actions:
            best_action = ("call", None)
    
    # Ensure valid action
    if best_action not in valid_actions:
        if ("check", None) in valid_actions and to_call == 0:
            best_action = ("check", None)
        elif ("fold", None) in valid_actions:
            best_action = ("fold", None)
        else:
            best_action = valid_actions[0]
    
    # Log decision time
    elapsed = (time.time() - start_time) * 1000  # ms
    # Uncomment for debugging
    # print(f"Decision time: {elapsed:.2f}ms, Action: {best_action}, Score: {best_score:.2f}")
    
    return best_action
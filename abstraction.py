# Global variables required for the abstraction functions
opponent_history = {"raises": 0, "calls": 0, "folds": 0}

def bucket_cards(hole_cards: list[str], community_cards: list[str], round_state: dict) -> str:
    """
    Card abstraction using ValueNetwork with street-based adjustments.
    Buckets: premium_hand, strong_hand, medium_hand, drawing_hand, weak_hand.
    """
    # Prepare round_state for ValueNetwork
    round_state_input = {
        'community_card': community_cards if community_cards else [],
        'action_histories': round_state.get('action_histories', {})
    }

    # Evaluate state using ValueNetwork
    with torch.no_grad():
        value = value_network(hole_cards, round_state_input).item()

    # Normalize value to [0, 1]
    value = (value - value_network.lin_final.bias.item()) / 10.0
    value = max(0.0, min(1.0, value))

    # Dynamic adjustment based on street
    street = round_state.get('street', 'preflop')
    street_modifiers = {
        'preflop': 0.9,  # Earlier streets require tighter play
        'flop': 1.0,
        'turn': 1.1,     # Later streets allow looser play
        'river': 1.2
    }
    modifier = street_modifiers.get(street, 1.0)
    adjusted_value = value * modifier
    adjusted_value = max(0.0, min(1.0, adjusted_value))

    # Define base thresholds
    thresholds = [0.8, 0.6, 0.5, 0.3]  # [premium, strong, medium, drawing]

    if adjusted_value > thresholds[0]:
        return "premium_hand"
    elif adjusted_value > thresholds[1]:
        return "strong_hand"
    elif adjusted_value > thresholds[2]:
        return "medium_hand"
    elif adjusted_value > thresholds[3]:
        return "drawing_hand"
    else:
        return "weak_hand"

def bucket_betting(pot_size: float, my_bet: float, opp_bet: float, blinds: float) -> str:
    """
    Betting abstraction considering pot size relative to blinds.
    Buckets: small_pot, medium_pot, large_pot, huge_pot.
    """
    total_pot = pot_size + my_bet + opp_bet
    pot_to_blinds = total_pot / blinds if blinds > 0 else float('inf')

    # Adjust thresholds based on blind levels
    thresholds = {
        'small': 5,   # Small pot: up to 5x blinds
        'medium': 10, # Medium pot: up to 10x blinds
        'large': 20   # Large pot: up to 20x blinds
    }

    if pot_to_blinds <= thresholds['small']:
        return "small_pot"
    elif pot_to_blinds <= thresholds['medium']:
        return "medium_pot"
    elif pot_to_blinds <= thresholds['large']:
        return "large_pot"
    else:
        return "huge_pot"

def model_opponent(opp_actions_history: dict, street: str) -> str:
    """
    Opponent modeling based on actions per street and historical data.
    Returns opponent type: "very_aggressive", "aggressive", "balanced", "passive", "very_passive", "unknown".
    """
    # Analyze actions for the current street
    street_actions = opp_actions_history.get(street, [])
    total_actions = len(street_actions)
    if total_actions == 0:
        return "unknown"

    # Calculate raise frequency for the current street
    raise_count = sum(1 for action in street_actions if action.get('action') == "raise")
    raise_freq = raise_count / total_actions

    # Combine with historical data
    total_historical = sum(opponent_history.values())
    historical_raise_freq = opponent_history["raises"] / total_historical if total_historical > 0 else 0

    # Weighted average of current and historical raise frequency
    combined_raise_freq = 0.7 * raise_freq + 0.3 * historical_raise_freq

    # Fine-grained classification
    if combined_raise_freq > 0.7:
        return "very_aggressive"
    elif combined_raise_freq > 0.5:
        return "aggressive"
    elif combined_raise_freq > 0.3:
        return "balanced"
    elif combined_raise_freq > 0.1:
        return "passive"
    else:
        return "very_passive"

import torch
from utils import SUITS, RANKS, SUIT_TO_INDEX, RANK_TO_INDEX

def encode_card(card):
    rank_str = str(card)[1]
    suit_str = str(card)[0]
    card_ind = SUIT_TO_INDEX[suit_str]*13 + RANK_TO_INDEX[rank_str]

    if suit_str not in SUITS or rank_str not in RANKS:
        raise ValueError(f"Invalid card string: {card}. "f"Suits must be in {SUITS}, ranks in {RANKS}.")
    
    # Return all indices +1 (the 0 index represents an empty)
    return SUIT_TO_INDEX[suit_str]+1, RANK_TO_INDEX[rank_str]+1, card_ind+1

def encode_bet(game_state, cur_player=0):
    # Encode opponent bets
    actions = game_state['table'].seats.players[1-cur_player].round_action_histories # Array of bets for the ["preflop","flop","turn","river"]
    streets = ["preflop","flop","turn","river"]
    
    # Maximum 6 bets per segment x 4 streets = 24 dim size
    actions_occured = [0 for i in range(6*len(streets))] 
    bet_sizes = [0 for i in range(6*len(streets))]
    for i in range(len(actions)):
        street_actions = actions[i]
        if street_actions == None:
            break

        for j in range(len(street_actions)):
            actions_occured[6*i + j] = 1 # Show that this bet has actually happened
            if street_actions[j]['action'] in ["FOLD","SMALLBLIND","BIGBLIND"]:
                bet_sizes[6*i + j] = 0
                continue
            bet_sizes[6*i + j] = street_actions[j]['paid'] # Get the new amount added to the pot
    
    return actions_occured, bet_sizes

def encode_game_state(hole_card, game_state, action, cur_player=0, device="cuda"):
    # Store hole cards (0 if no card in this slot)
    hole_suit = [0 for i in range(2)]
    hole_rank = [0 for i in range(2)]
    hole_card_idx = [0 for i in range(2)]
    for i, card in enumerate(hole_card):
        hole_suit[i], hole_rank[i], hole_card_idx[i] = encode_card(card)

    # Store the board cards (0 if no card in this slot)
    board_suit = [0 for i in range(5)]
    board_rank = [0 for i in range(5)]
    board_card_idx = [0 for i in range(5)]
    for i, card in enumerate(game_state['table']._community_card):
        board_suit[i], board_rank[i], board_card_idx[i] = encode_card(card)

    actions_occured, bet_sizes = encode_bet(game_state, cur_player) # Store bets
    actions = ["fold", "call", "raise"]                             # Store action
    action_num = actions.index(action['action'])

    # Convert to tensors
    hole_suit = torch.tensor(hole_suit, dtype=torch.int).to(device)
    hole_rank = torch.tensor(hole_rank, dtype=torch.int).to(device)
    hole_card_idx = torch.tensor(hole_card_idx, dtype=torch.int).to(device)
    board_suit = torch.tensor(board_suit, dtype=torch.int).to(device)
    board_rank = torch.tensor(board_rank, dtype=torch.int).to(device)
    board_card_idx = torch.tensor(board_card_idx, dtype=torch.int).to(device)
    actions_occured = torch.tensor(actions_occured, dtype=torch.float).to(device)
    bet_sizes = torch.tensor(bet_sizes, dtype=torch.float).to(device)
    action_num = torch.tensor(action_num, dtype=torch.int).to(device)
    return hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes, action_num
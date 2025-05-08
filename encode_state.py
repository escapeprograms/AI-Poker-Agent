import torch
from utils import SUITS, RANKS, SUIT_TO_INDEX, RANK_TO_INDEX

def encode_card(card):

    num_suits = len(SUITS)
    num_ranks = len(RANKS)

    rank_str = str(card)[1]
    suit_str = str(card)[0]
    card_ind = SUIT_TO_INDEX[suit_str]*13 + RANK_TO_INDEX[rank_str]

    if suit_str not in SUITS or rank_str not in RANKS:
        raise ValueError(f"Invalid card string: {card}. "
                            f"Suits must be in {SUITS}, ranks in {RANKS}.")
    
    #return all indices +1 (the 0 index represents an empty)
    return SUIT_TO_INDEX[suit_str]+1, RANK_TO_INDEX[rank_str]+1, card_ind+1

def encode_bet(round_state, uuid):
    actions = round_state['action_histories'] #dictionary with keys ["preflop","flop","turn","river"]
    streets = ["preflop","flop","turn","river"]
    
    actions_occured = [0 for i in range(6*len(streets))] #maximum 6 bets per segment x 4 streets = 24 dim size
    bet_sizes = [0 for i in range(6*len(streets))]
    for i in range(len(actions)):
        street_actions = actions[streets[i]]
        for j in range(len(street_actions)):
            actions_occured[6*i + j] = 1 #show that this bet has actually happened
            # print("STREET ACTION", street_actions[j])
            if street_actions[j]['action'] in ["FOLD","SMALLBLIND","BIGBLIND"]:
                bet_sizes[6*i + j] = 0
                continue

            #only record opponent bets
            if street_actions[j]['uuid'] == uuid:
                continue
            bet_sizes[6*i + j] = street_actions[j]['paid'] #get the new amount added to the pot
    
    return actions_occured, bet_sizes

def encode_game_state(hole_card, round_state, uuid):
    #store hoel cards (0 if no card in this slot)
    hole_suit = [0 for i in range(2)]
    hole_rank = [0 for i in range(2)]
    hole_card_idx = [0 for i in range(2)]
    for i, card in enumerate(hole_card):
        hole_suit[i], hole_rank[i], hole_card_idx[i] = encode_card(card)

    #store the board cards (0 if no card in this slot)
    board_suit = [0 for i in range(5)]
    board_rank = [0 for i in range(5)]
    board_card_idx = [0 for i in range(5)]
    for i, card in enumerate(round_state['community_card']):
        board_suit[i], board_rank[i], board_card_idx[i] = encode_card(card)

    #store bets
    actions_occured, bet_sizes = encode_bet(round_state, uuid)
    # hole_suit = torch.tensor(hole_suit, dtype=torch.int64)
    # hole_rank = torch.tensor(hole_rank, dtype=torch.int64)
    # hole_card_idx = torch.tensor(hole_card_idx, dtype=torch.int64)
    # board_suit = torch.tensor(board_suit, dtype=torch.int64)
    # board_rank = torch.tensor(board_rank, dtype=torch.int64)
    # board_card_idx = torch.tensor(board_card_idx, dtype=torch.int64)
    # actions_occured = torch.tensor(actions_occured, dtype=torch.int64)
    # bet_sizes = torch.tensor(bet_sizes, dtype=torch.float32)
    return hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes
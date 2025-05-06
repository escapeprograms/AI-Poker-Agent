import torch
import torch.nn as nn

class CardEmbedding(nn.Module):
    def __init__(self, embedding_dim=64):
        super(CardEmbedding, self).__init__()

        # Define the suits and ranks
        self.suits = ['C', 'D', 'H', 'S']  # Clubs, Diamonds, Hearts, Spades
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

        self.num_suits = len(self.suits)
        self.num_ranks = len(self.ranks)
        self.embedding_dim = embedding_dim

        # Embeddings for suit, rank, and individual card
        self.suit_embedding = nn.Embedding(self.num_suits, embedding_dim)
        self.rank_embedding = nn.Embedding(self.num_ranks, embedding_dim)
        self.card_embedding = nn.Embedding(self.num_ranks*self.num_suits, embedding_dim)


        # Create mappings from card strings to indices
        self.suit_to_index = {suit: i for i, suit in enumerate(self.suits)}
        self.rank_to_index = {rank: i for i, rank in enumerate(self.ranks)}

    def forward(self, cards):
        suit_indices = []
        rank_indices = []
        card_indices = []
        for card in cards:
            rank_str = card[1]
            suit_str = card[0]
            card_ind = self.suit_to_index[suit_str]*13 + self.rank_to_index[rank_str]

            if suit_str not in self.suits or rank_str not in self.ranks:
                raise ValueError(f"Invalid card string: {card}. "
                                 f"Suits must be in {self.suits}, ranks in {self.ranks}.")
            suit_indices.append(self.suit_to_index[suit_str])
            rank_indices.append(self.rank_to_index[rank_str])
            card_indices.append(card_ind)

        suit_indices = torch.tensor(suit_indices, dtype=torch.long)
        rank_indices = torch.tensor(rank_indices, dtype=torch.long)
        card_indices = torch.tensor(card_indices, dtype=torch.long)

        suit_embedding = self.suit_embedding(suit_indices)
        rank_embedding = self.rank_embedding(rank_indices)
        card_embedding = self.card_embedding(card_indices)
        
        final_embeddings = suit_embedding + rank_embedding + card_embedding
        return torch.sum(final_embeddings, dim=0) #create an embedding for the hand by summing embeddings for the cards

#building block w/ skip connection and relu activation
class SkipRelu(nn.Module):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.lin = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        input = x
        out = self.lin(x)
        out = self.relu(out)
        out = out + input
        return out

class ValueNetwork(nn.Module):
    def __init__(self, embedding_dim = 64, hidden_size = 192):
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.card_embedder = CardEmbedding(self.embedding_dim)

        self.relu = nn.ReLU()
        self.lin_skip_small = SkipRelu(self.embedding_dim)
        self.lin_skip_large = SkipRelu(self.hidden_size)

        
        self.merge_cards = nn.Linear(self.embedding_dim*2, self.hidden_size)
        self.shrink_cards = nn.Linear(self.hidden_size, embedding_dim)

        self.encode_bets = nn.Linear(48, self.embedding_dim)

        self.merge_main = nn.Linear(self.embedding_dim*2, self.embedding_dim)

        self.lin_final = nn.Linear(self.embedding_dim, 1)

    def forward(self, hole, round_state): #takes in the inputs from the entire state
        hand_embedding = self.card_embedder(hole) #hand cards
        board_embedding = self.card_embedder(round_state['community_card']) #board cards
        
        #card portion of the network
        hand = self.lin_skip_small(hand_embedding)
        board = self.lin_skip_small(board_embedding)

        cards_layer = torch.cat(hand, board)
        cards_layer = self.merge_cards(cards_layer)
        cards_layer = self.relu(cards_layer)

        cards_layer = self.lin_skip_large(cards_layer)
        cards_layer = self.lin_skip_large(cards_layer)
        cards_layer = self.shrink_cards(cards_layer)
        cards_layer = self.relu(cards_layer)

        #bets/actions portion of the network
        actions = round_state['action_histories']
        streets = ["preflop","flop","turn","river"]
        
        actions_occured = [0 for i in range(6*len(streets))] #maximum 6 bets per segment x 4 streets = 24 dim size
        bet_sizes = [0 for i in range(6*len(streets))]
        for i in range(len(streets)):
            street_actions = actions[streets[i]]
            for j in range(len(street_actions)):
                actions_occured[6*i + j] = 1 #show that this bet has actually happened
                bet_sizes[6*i + j] = street_actions[j]['add_amount'] #get the new amount added to the pot

        actions_occured = torch.tensor(actions_occured, dtype=torch.long)
        bet_sizes = torch.tensor(bet_sizes, dtype=torch.long)
        bets_layer = torch.cat(actions_occured, bet_sizes) #24+24=48 dim size

        bets_layer = self.encode_bets(bets_layer)
        bets_layer = self.relu(bets_layer)
        bets_layer = self.lin_skip_small(bets_layer)

        #combine
        main_layer = torch.cat(cards_layer, bets_layer)
        main_layer = self.merge_cards(main_layer)
        main_layer = self.lin_skip_small(main_layer)
        main_layer = self.lin_skip_small(main_layer)
        main_layer = self.lin_final(main_layer)
        return main_layer #return a single value
        

if __name__ == '__main__':
    embedding_dim = 32
    card_embedder = CardEmbedding(embedding_dim)

    # Example usage
    hand = ['SA', 'HK', 'C2', 'DT']
    embeddings = card_embedder(hand)

    print(f"Input hand: {hand}")
    print(f"Shape of embeddings: {embeddings.shape}")
    print(f"Embeddings:\n{embeddings}")

    # Example with a single card
    single_card = ['CQ']
    single_embedding = card_embedder(single_card)
    print(f"\nInput card: {single_card}")
    print(f"Shape of single embedding: {single_embedding.shape}")
    print(f"Single embedding:\n{single_embedding}")
import torch
import torch.nn as nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CardEmbedding(nn.Module):
    def __init__(self, embedding_dim=64):
        super(CardEmbedding, self).__init__()
        
        # Define the suits and ranks
        self.suits = ['C', 'D', 'H', 'S']  # Clubs, Diamonds, Hearts, Spades
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

        self.num_suits = len(self.suits) + 1 #add one for an empty dummy
        self.num_ranks = len(self.ranks) + 1
        self.embedding_dim = embedding_dim

        # Embeddings for suit, rank, and individual card
        self.suit_embedding = nn.Embedding(self.num_suits, embedding_dim)
        self.rank_embedding = nn.Embedding(self.num_ranks, embedding_dim)
        self.card_embedding = nn.Embedding(self.num_ranks*self.num_suits, embedding_dim) #note: there are some extra 0s 


        # Create mappings from card st

    def forward(self, suit_indices, rank_indices, card_indices):

        # suit_indices = torch.tensor(suit_indices, dtype=torch.long).to(device)
        # rank_indices = torch.tensor(rank_indices, dtype=torch.long).to(device)
        # card_indices = torch.tensor(card_indices, dtype=torch.long).to(device)

        suit_embedding = self.suit_embedding(suit_indices)
        rank_embedding = self.rank_embedding(rank_indices)
        card_embedding = self.card_embedding(card_indices)
        
        final_embedding = suit_embedding + rank_embedding + card_embedding
        return final_embedding

#building block w/ skip connection and relu activation
class SkipRelu(nn.Module):
    def __init__(self, hidden_size):
        super(SkipRelu, self).__init__()

        self.hidden_size = hidden_size
        self.lin = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        input = x
        out = self.lin(x)
        out = self.relu(out)
        out = out + input
        return out

#Value Network returns 2 things: a value for the state and a regret for the state
class ValueNetwork(nn.Module):
    def __init__(self, embedding_dim = 64, hidden_size = 192):
        super(ValueNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.card_embedder = CardEmbedding(self.embedding_dim)

        self.relu = nn.ReLU()
        self.lin_skip_small = SkipRelu(self.embedding_dim)
        self.lin_skip_large = SkipRelu(self.hidden_size)

        
        self.merge_cards = nn.Linear(self.embedding_dim*2, self.hidden_size)
        self.shrink_cards = nn.Linear(self.hidden_size, embedding_dim)

        self.encode_action = nn.Embedding(3, self.embedding_dim)
        self.encode_bets = nn.Linear(48, self.embedding_dim)

        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.merge_main = nn.Linear(self.embedding_dim*3, self.embedding_dim)

        self.lin_final = nn.Linear(self.embedding_dim, 1)

    def forward(self, hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes, action): #takes in the inputs from the entire state
        hand_embedding_all_cards = self.card_embedder(hole_suit, hole_rank, hole_card_idx) #hand cards
        hand_embedding = torch.sum(hand_embedding_all_cards, dim=-2)

        board_embedding_all_cards = self.card_embedder(board_suit, board_rank, board_card_idx) #board cards
        board_embedding = torch.sum(board_embedding_all_cards, dim=-2)
        
        #card portion of the network
        hand = self.lin_skip_small(hand_embedding)
        board = self.lin_skip_small(board_embedding)
        cards_layer = torch.cat((hand, board), dim=-1)
        cards_layer = self.merge_cards(cards_layer)
        cards_layer = self.relu(cards_layer)

        cards_layer = self.lin_skip_large(cards_layer)
        # cards_layer = self.lin_skip_large(cards_layer)
        cards_layer = self.shrink_cards(cards_layer)
        cards_layer = self.relu(cards_layer)

        #action portion of the network
        # action = torch.tensor(action, dtype=torch.int32).to(device)
        action_layer = self.encode_action(action)
        action_layer = self.lin_skip_small(action_layer)

        #bets/actions portion of the network
        # actions_occured = torch.tensor(actions_occured, dtype=torch.float).to(device)
        # bet_sizes = torch.tensor(bet_sizes, dtype=torch.float).to(device)
        bets_layer = torch.cat((actions_occured, bet_sizes), dim=-1) #24+24=48 dim size

        bets_layer = self.encode_bets(bets_layer)
        bets_layer = self.relu(bets_layer)
        bets_layer = self.lin_skip_small(bets_layer)

        #combine these components in 2 ways: a value layer and a regret layer
        regret_layer = torch.cat((cards_layer, bets_layer, action_layer), dim=-1) #2 * small layer
        regret_layer = self.merge_main(regret_layer)
        regret_layer = self.lin_skip_small(regret_layer)
        regret_layer = self.layer_norm(regret_layer)
        regret_layer = self.lin_skip_small(regret_layer)
        regret_layer = self.lin_final(regret_layer)

        value_layer = torch.cat((cards_layer, bets_layer, action_layer), dim=-1) #2 * small layer
        value_layer = self.merge_main(value_layer)
        value_layer = self.lin_skip_small(value_layer)
        value_layer = self.layer_norm(value_layer)
        value_layer = self.lin_skip_small(value_layer)
        value_layer = self.lin_final(value_layer)



        return torch.cat((regret_layer, value_layer), dim=-1) #return the value and regret
        

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
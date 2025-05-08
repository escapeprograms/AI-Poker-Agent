import torch
import torch.nn as nn
import numpy as np
from utils import expectedValue, SUITS, RANKS, CARDS, example_policy, random_policy
from encode_state import encode_card


class PolicyNet(nn.Module):
    def __init__(self, embedding_dim=32):
        super(PolicyNet, self).__init__()

        # Define the suits and ranks
        self.suits = SUITS  
        self.ranks = RANKS

        self.num_suits = len(self.suits)
        self.num_ranks = len(self.ranks)
        self.embedding_dim = embedding_dim

        # Embeddings for suit, rank, and individual card
        self.suit_embedding = nn.Embedding(self.num_suits+1, embedding_dim)
        self.rank_embedding = nn.Embedding(self.num_ranks+1, embedding_dim)
        self.card_embedding = nn.Embedding(self.num_ranks*self.num_suits+1, embedding_dim)


        # Create mappings from card strings to indices
        self.suit_to_index = {suit: i for i, suit in enumerate(self.suits)}
        self.rank_to_index = {rank: i for i, rank in enumerate(self.ranks)}

        # Create layers
        self.fc1 = nn.Linear(embedding_dim+2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3) # fold, call, raise
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, suits, ranks, card_ids, pot, canRaise):
        suit_indices = torch.tensor(suits, dtype=torch.int)
        rank_indices = torch.tensor(ranks, dtype=torch.int)
        card_indices = torch.tensor(card_ids, dtype=torch.int)

        suit_embedding = self.suit_embedding(suit_indices)
        rank_embedding = self.rank_embedding(rank_indices)
        card_embedding = self.card_embedding(card_indices)
        
        final_embeddings = suit_embedding + rank_embedding + card_embedding # Embeddings of each card in a 'list'
        hand_embeddings = torch.sum(final_embeddings, dim=0) # Create an embedding for the hand by summing embeddings for the cards

        x = torch.cat((hand_embeddings, torch.tensor((pot, canRaise))))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        # Note that x is already nonnegative because of sigmoid
        x = x / x.sum() # Normalize so sum is 1
        return x

def model_wrapper(model: PolicyNet):
    def policy(hand: list, community: list, pot: int, canRaise: bool):
        cards = hand+community
        model.eval()
        suits, ranks, cardIds = [], [], []
        for card in cards:
            s, r, c = encode_card(card)
            suits.append(s)
            ranks.append(r)
            cardIds.append(c)
        output = model(suits, ranks, cardIds, pot, int(canRaise))
        # We assume that the output is already nonnegative!
        if not canRaise:
            output[2] = 0
        output = output / output.sum()
        return output.tolist()
    return policy

def testGame(model, model2=None, dataset=None):
    policy = model # Don't automatically wrap anymore
    policy2 = policy if model2 is None else model2 #model_wrapper(model2)
    cards = list(np.random.choice(CARDS, 9, False)) # Generate random cards
    hand1 = cards[:2]
    hand2 = cards[2:4]
    community = cards[4:]
    stats = expectedValue(policy, policy2, hand1, hand2, community, fullStats=True, dataset=dataset)
    ev, f, c, r = stats
    # print(ev, f, c, r)#
    return stats



# #Run self-play to gather data, then train the value function
# num_epochs = 20
# batch_size = 1
# num_rounds = 32
# explore_chance = 1

# for j in range(30):
#     state = simulate(evaluation_function, num_rounds = num_rounds, explore_chance=explore_chance)
#     train_loop(*state, evaluation_function, num_epochs=num_epochs, batch_size=batch_size)
#     evaluation_function.eval()

#     explore_chance *= 0.95
#     if num_rounds < 100000:
#         num_rounds *= 2

#     #save model
#     torch.save(evaluation_function.state_dict(), "models/evaluation_function.pth")

def testN(model, baseline, n):
    res = np.zeros(4)
    for i in range(n):
        res = res + np.array(testGame(model, baseline))
    return res/n

def comp(model, baseline, n, name):
    res = testN(model, baseline, n)
    print(f"As first player against {name}:")
    print(res)
    res = testN(baseline, model, n)
    print(f"As second player against {name}:")
    print(-res)
    print()

def test_suite(model, n):
    # Baseline
    baseline = PolicyNet()
    baseline.load_state_dict(torch.load("models/evaluation_function[latest-test].pth"))

    raise_policy = lambda a, b, c, canRaise: (0, 0, 1) if canRaise else (0, 1, 0)
    call_policy = lambda a, b, c, d: (0, 1, 0)
    fold_policy = lambda a, b, c, d: (1, 0, 0)

    comp(model, random_policy, n, "random")
    comp(model, example_policy, n, "uniform")
    comp(model, model_wrapper(baseline), n, "baseline")
    comp(model, raise_policy, n, "raise")
    comp(model, call_policy, n, "call")
    comp(model, fold_policy, n, "fold")


CURR_MODEL = PolicyNet()
# CURR_MODEL.load_state_dict(torch.load("models/policy[v2].pth"))
CURR_MODEL.load_state_dict(torch.load("models/evaluation_function[latest-test].pth"))

if __name__ == '__main__':

    model2 = PolicyNet()
    model2.load_state_dict(torch.load("models/policy[v2].pth"))

    test_suite(model_wrapper(model2), 5)
    # dataset = None

    # N = 10

    # res = np.zeros(4)
    # for i in range(N):
    #     res = res + np.array(testGame(model_wrapper(model2), random_policy, dataset=dataset))
    # print("As first player against random: ", res/N)
    # # -22.079 --> -12.639

    # res = np.zeros(4)
    # for i in range(N):
    #     res = res + np.array(testGame(random_policy, model_wrapper(model2), dataset=dataset))
    # print("As second player against random: ", res/N)
    # #27.123 --> 22.926

    # res = np.zeros(4)
    # for i in range(N):
    #     res = res + np.array(testGame(model_wrapper(model2), example_policy, dataset=dataset))
    # print("As first player against uniform: ", res/N)
    # #9.551 --> 3.485

    # res = np.zeros(4)
    # for i in range(N):
    #     res = res + np.array(testGame(example_policy, model_wrapper(model2), dataset=dataset))
    # print("As second player against uniform: ", res/N)
    # #-10.659 --> -5.055

    # res = np.zeros(4)
    # for i in range(N):
    #     res = res + np.array(testGame(model_wrapper(model2), model_wrapper(model), dataset=dataset))
    # print("As first player against baseline: ", res/N)
    # #-15.127

    # res = np.zeros(4)
    # for i in range(N):
    #     res = res + np.array(testGame(model_wrapper(model), model_wrapper(model2), dataset=dataset))
    # print("As second player against baseline: ", res/N)
    # #1.434


    
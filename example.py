from pypokerengine.api.game import setup_config, start_poker
from minimax_player import MinimaxPlayer
from raise_player import RaisedPlayer
from randomplayer import RandomPlayer
from value_model import CardEmbedding, ValueNetwork
import torch

config = setup_config(max_round=1, initial_stack=10000, small_blind_amount=10)

with torch.no_grad():
    evaluation_function = ValueNetwork()
    evaluation_function.load_state_dict(torch.load("models/evaluation_function.pth", weights_only=True))
    evaluation_function.eval()

    config.register_player(name="Eric", algorithm=MinimaxPlayer(evaluation_function))
    config.register_player(name="Evil", algorithm=RaisedPlayer())

    game_result = start_poker(config, verbose=1)

print(game_result)


#raise limitations: each player can only raise 4 times in total, and there can only be 4 raises in a street

print(game_result)


#raise limitations: each player can only raise 4 times in total, and there can only be 4 raises in a street
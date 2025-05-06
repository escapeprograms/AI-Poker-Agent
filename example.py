from pypokerengine.api.game import setup_config, start_poker
from test_player import TestPlayer
from raise_player import RaisedPlayer
from randomplayer import RandomPlayer

config = setup_config(max_round=1, initial_stack=10000, small_blind_amount=10)

config.register_player(name="Eric", algorithm=TestPlayer())
config.register_player(name="Evil", algorithm=RaisedPlayer())

game_result = start_poker(config, verbose=1)

print(game_result)


#raise limitations: each player can only raise 4 times in total, and there can only be 4 raises in a street

print(game_result)


#raise limitations: each player can only raise 4 times in total, and there can only be 4 raises in a street
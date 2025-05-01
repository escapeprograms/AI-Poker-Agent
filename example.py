from pypokerengine.api.game import setup_config, start_poker
from test_player import TestPlayer
from raise_player import RaisedPlayer
from randomplayer import RandomPlayer

config = setup_config(max_round=10, initial_stack=10000, small_blind_amount=10)

config.register_player(name="Test", algorithm=TestPlayer())
config.register_player(name="Raiser", algorithm=RaisedPlayer())

game_result = start_poker(config, verbose=1)

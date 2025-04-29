from pypokerengine.api.game import setup_config, start_poker
from agent.test_player import TestPlayer
from raise_player import RaisedPlayer
from randomplayer import RandomPlayer

#TODO: config the config as our wish
config = setup_config(max_round=1, initial_stack=10000, small_blind_amount=10)

config.register_player(name="f1", algorithm=TestPlayer())
config.register_player(name="FT2", algorithm=RaisedPlayer())

game_result = start_poker(config, verbose=1)

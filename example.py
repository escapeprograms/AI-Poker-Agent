from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer, CallPlayer
from rebel_player import RebelPlayer

#TODO:config the config as our wish
config = setup_config(max_round=1, initial_stack=10000, small_blind_amount=10)



config.register_player(name="f1", algorithm=CallPlayer())
config.register_player(name="FT2", algorithm=RaisedPlayer())


game_result = start_poker(config, verbose=1)

print(game_result)


#raise limitations: each player can only raise 4 times in total, and there can only be 4 raises in a street
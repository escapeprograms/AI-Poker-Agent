from tqdm import tqdm
from pypokerengine.api.game import setup_config, start_poker
from minimax_player import MinimaxPlayer
from raise_player import RaisedPlayer, CallPlayer
from randomplayer import RandomPlayer
from CFRD_player import CFRDPlayer
from value_model import CardEmbedding, ValueNetwork
from custom_player import CustomPlayer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


models = ["submission/models/CFR-D_cp9.pth", "submission/models/CFR-D_cp8.pth", "submission/models/CFR-D_cp7.pth", 
          "submission/models/CFR-D_cp6.pth", "submission/models/CFR-D_cp5.pth", 
          "submission/models/CFR-D_cp4.pth", "submission/models/CFR-D_cp3.pth", "submission/models/CFR-D_cp2.pth", 
          "submission/models/CFR-D_cp1.pth", "submission/models/CFR-D_cp0.pth"]
def run_game(algo1, algo2):
    config = setup_config(max_round=500, initial_stack=1000, small_blind_amount=10)
    config.register_player(name="Eric", algorithm=algo1)
    config.register_player(name="Evil", algorithm=algo2)

    game_result = start_poker(config, verbose=0)
    
    winner = None
    for i in range(len(game_result['players'])):
        if game_result['players'][i]['state'] == 'participating':
            winner = i
    return winner

results = [0 for m in models]

with torch.no_grad():
    for i, model in enumerate(models):
        for j in tqdm(range(100)):
            evaluation_function = ValueNetwork()
            evaluation_function.load_state_dict(torch.load(model, weights_only=True))
            evaluation_function.to(device)
            evaluation_function.eval()

            cfrd = CFRDPlayer(evaluation_function, device="cuda")
            raised = RaisedPlayer()

            winner = run_game(cfrd, raised)
            if winner == 0:
                results[i] += 1
        print("Model:", model," Wins:", results[i])
    
print("Results:", results)
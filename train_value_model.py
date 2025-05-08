import random
from pypokerengine.api.emulator import Emulator
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.utils.game_state_utils import attach_hole_card, attach_hole_card_from_deck, restore_game_state
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.round_manager import RoundManager
from pypokerengine.engine.action_checker import ActionChecker
from encode_state import encode_game_state
from minimax_player import MinimaxPlayer
from raise_player import RaisedPlayer
from randomplayer import RandomPlayer
from value_model import ValueNetwork
from training.value_dataset import ValueDataset

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_device = "cpu"

#partial game traversal
initial_stack = 10000

def traverse(training_data, game_state, events, depth=3, value_network = None, training_round = 0):
    cur_player = game_state["next_player"]
    # round_state = events[-1][1]["message"]["round_state"]
    hole_card = game_state['table'].seats.players[cur_player].hole_card
    uuid = game_state['table'].seats.players[cur_player].uuid
    
    # print(round_state)
    #game is over
    if (game_state["street"] == Const.Street.FINISHED):
        # winners = events[-1][1]["message"]["winners"]
        # print("winner",winners)
        # is this wrong? do we need to make it the same for both players?
        
        # print("I am player", cur_player, "with hole card", hole_card)
        # print("my value (terminal)", game_state['table'].seats.players[cur_player].stack - initial_stack)
        return game_state['table'].seats.players[cur_player].stack - initial_stack #return actual payoff
    
    # print("bets for", cur_player, game_state['table'].seats.players[cur_player].round_action_histories)
    # Generate legal actions at current state
    actions = ActionChecker.legal_actions(
        game_state["table"].seats.players, 
        game_state["next_player"],
        game_state["small_blind_amount"],
        game_state["street"]
    )

    #calculate a policy
    policy = []
    pred_vals = []
    for action in actions:
        #calculate the value of each action
        if training_round==0: #for the first round, force a uniform distribution
            val = 0
        else:
            val = value_network(*encode_game_state(hole_card, game_state, action, cur_player, device=eval_device)).item()
        pred_vals.append(max(0, val)) #don't allow negative values
    total_val = sum(pred_vals)
    for val in pred_vals:
        if total_val == 0:
            probability = 1 / len(pred_vals) #uniform distribution if no info yet
        else:
            probability = val / total_val
        policy.append(probability)
    
    #choose an action to take
    action_indices = list(range(len(policy)))
    selected_action = random.choices(action_indices, weights=policy, k=1)[0]

    # Search actions recursively
    values = [] #store values of each action
    for i, action in enumerate(actions):
        next_state, events = RoundManager.apply_action(game_state, action["action"])
        # sample 1 action to traverse, and approximate the rest with the value network
        if i == selected_action or next_state["street"] == Const.Street.FINISHED: #also get the true value if its close
            #since we are getting the opponents expected value, we negate
            values.append(-traverse(training_data, next_state, events, depth - 1, value_network))
        else:
            values.append(pred_vals[i]) #note: depth is unused

        #DEPRECATED: this was a full tree exploration
        # #stop traversing and approximate at depth 0
        # if depth == 0:
        #     values.append(pred_vals[i])
        #     continue

        # #recursively traverse
        # values.append(traverse(training_data, next_state, events, depth - 1, value_network))

    #calculate regret of each action
    expected_value = np.dot(np.array(policy), np.array(values)) #weighted average of traversed values
    
    # print("I am player", cur_player, "with hole card", hole_card)
    # print("my values", values)
    # print("my policy", policy)
    # print("expected value", expected_value)
    for i, action in enumerate(actions):
        regret = values[i] - expected_value #instantaneous regret
        #add a training example
        inputs = encode_game_state(hole_card, game_state, action, cur_player) #a list of all the things
        for i, input in enumerate(inputs):
            training_data[i].append(input)
        
        training_data[-1].append(torch.tensor(regret, dtype=torch.float))


    #return the expected value at this state
    return expected_value.item() #convert np float to normal number

def simulate(evaluation_function, num_rounds=3200, training_round=0):
    # 1. Initialize the emulator
    emulator = Emulator()

    # 2. Set the game rules
    emulator.set_game_rule(player_num=2, max_round=10, small_blind_amount=10, ante_amount=0)

    # 3. Define player information
    players_info = {
        "player1": {"name": "Player 1", "stack": initial_stack},
        "player2": {"name": "Player 2", "stack": initial_stack},
    }

    training_data = [[] for i in range(10)]
    for K in tqdm(range(num_rounds)):
        initial_game_state = emulator.generate_initial_game_state(players_info)
        game_state, events = emulator.start_new_round(initial_game_state)
        traverse(training_data, game_state, events, 10, evaluation_function, training_round=training_round)
    
    return training_data
    #extract training data from both players
    # hole_suit1, hole_rank1, hole_card_idx1, board_suit1, board_rank1, board_card_idx1, actions_occured1, bet_sizes1 = None
    # hole_suit2, hole_rank2, hole_card_idx2, board_suit2, board_rank2, board_card_idx2, actions_occured2, bet_sizes2 = None
    # values = p1.train_values + p2.train_values
    # return hole_suit1 + hole_suit2, hole_rank1 + hole_rank2, hole_card_idx1 + hole_card_idx2,\
    #       board_suit1 + board_suit2, board_rank1 + board_rank2, board_card_idx1 + board_card_idx2,\
    #       actions_occured1 + actions_occured2, bet_sizes1 + bet_sizes2, values


########

evaluation_function = ValueNetwork()
evaluation_function.to(eval_device)


def train_loop(hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes, action, values, model, num_epochs=30, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    dataset = ValueDataset(hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes, action, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Set the model to training mode
    model.train()
    print("# of training data:", len(dataset))
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes, action, value) in enumerate(dataloader):
            hole_suit.to(device)
            hole_rank.to(device)
            hole_card_idx.to(device)
            board_suit.to(device)
            board_rank.to(device)
            board_card_idx.to(device)
            actions_occured.to(device)
            bet_sizes.to(device)
            action.to(device)
            value.to(device)
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: Get the predicted values for the current states
            predicted_values = model(hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes, action)


            # Calculate the loss
            value = value.unsqueeze(1) 
            loss = criterion(predicted_values.to(device), value.to(device))

            # Backward pass: Compute gradients
            loss.backward()

            # Optimization step: Update the model's weights
            optimizer.step()

            running_loss += loss.item()

        # Print statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss/len(dataset):.4f}")
        running_loss = 0.0

    print("Finished Training")

#Run self-play to gather data, then train the value function
num_epochs = 30
batch_size = 32
num_rounds = 1000

for j in range(10):
    print("running round", j)
    #re-initialize model after simulating
    

    state = simulate(evaluation_function, num_rounds = num_rounds, training_round = j)
    evaluation_function = ValueNetwork()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(evaluation_function.parameters(), lr=0.0001)

    evaluation_function.to(device)
    train_loop(*state, evaluation_function, num_epochs=num_epochs, batch_size=batch_size)
    evaluation_function.eval()
    evaluation_function.to(eval_device) #evaluate on eval device (CPU)

    # explore_chance *= 0.95
    # if num_rounds < 100000:
    #     num_rounds *= 2

    #save model
    torch.save(evaluation_function.state_dict(), "models/evaluation_function_regret.pth")
import random
import sys
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.poker_constants import PokerConstants as Const
from pypokerengine.engine.round_manager import RoundManager
from pypokerengine.engine.action_checker import ActionChecker
from encode_state import encode_game_state
from value_model import ValueNetwork
from training.value_dataset import ValueDataset
from training.data_offloader import store_training_data, load_training_data

from CFRD_player import CFRDPlayer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import gc
import os

#hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_device = "cpu"

#supervised learning
num_epochs = 10
batch_size = 64
num_rounds = 50000 #number of rounds to simulate per iteration
num_iterations = 100 # number of self-play and training iterations

save_data_segments = False #
segment_size = 50000 #how often to save data

initial_stack = 10000

verbose_print = False


#partial game traversal
def traverse(training_data, game_state, events, value_network = None, training_round = 0):
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
        
        if verbose_print == True:
            print("I am player", cur_player)
            print("my value (terminal)", game_state['table'].seats.players[cur_player].stack - initial_stack)
        return game_state['table'].seats.players[cur_player].stack - initial_stack #return actual payoff
    
    # print("bets for", cur_player, game_state['table'].seats.players[cur_player].round_action_histories)
    # Generate legal actions at current state
    actions = ActionChecker.legal_actions(
        game_state["table"].seats.players, 
        game_state["next_player"],
        game_state["small_blind_amount"],
        game_state["street"]
    )

    #predict regrets/advantages for each action
    pred_vals = []
    policy = []
    if training_round==0: #for the first round, force a distribution similar to random_player with less of a chance of folding
        if (len(actions) == 2):
            pred_vals = [0, 0]
            policy = [0.03, 0.97]
        elif (len(actions) == 3):
            pred_vals = [0, 0, 0]
            policy = [0.03, 0.49, 0.48]
    else:
        pred_vals, policy = CFRDPlayer.get_policy(game_state, hole_card, actions, value_network, cur_player)
    
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
            values.append(-traverse(training_data, next_state, events, value_network))
        else:
            values.append(pred_vals[i]) #sample the value network

    #calculate regret of each action
    expected_value = np.dot(np.array(policy), np.array(values)) #weighted average of traversed values (expected counterfactual value)
    
    if verbose_print == True:
        print("I am player", cur_player, "with hole card", str(hole_card[0]), str(hole_card[1]))
        community_cards = [str(card) for card in game_state['table']._community_card]
        print("community cards", community_cards)
        print("my values", values)
        print("my policy", policy)
        print("expected value", expected_value)
    
    #create a training example for each action
    for i, action in enumerate(actions):
        regret = values[i] - expected_value #instantaneous regret
        #add a training example
        inputs = encode_game_state(hole_card, game_state, action, cur_player, device="cpu") #a list of all the things
        for j, input in enumerate(inputs):
            training_data[j].append(input)
        
        training_data[-2].append(torch.tensor(regret, dtype=torch.float)) #record action regret
        training_data[-1].append(torch.tensor(values[i], dtype=torch.float)) #record action value


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

    training_data = [[] for i in range(11)]
    for K in tqdm(range(num_rounds)):
        initial_game_state = emulator.generate_initial_game_state(players_info)
        game_state, events = emulator.start_new_round(initial_game_state)
        traverse(training_data, game_state, events, evaluation_function, training_round=training_round)

        #save training data in chunks
        if save_data_segments and K % segment_size == segment_size - 1:
            store_training_data(training_data, K//segment_size)
            for data in training_data:
                del data
            del training_data
            gc.collect()

            training_data = [[] for i in range(11)] #reset ram
            
    
    if save_data_segments:
        #add residual training data
        final_segment = num_rounds//segment_size
        stored_training_data = load_training_data(final_segment + 1)
        for i in range(len(training_data)):
            training_data[i] += stored_training_data[i]
            del stored_training_data[i] #free memory
            
        
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

# simulate(evaluation_function, num_rounds=1, training_round=0)

def train_loop(hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes, action, regret, value, model, num_epochs=30, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    dataset = ValueDataset(hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes, action, regret, value)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Set the model to training mode
    model.train()
    print("# of training data:", len(dataset))
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes, action, regret, value) in enumerate(dataloader):
            hole_suit = hole_suit.to(device)
            hole_rank = hole_rank.to(device)
            hole_card_idx = hole_card_idx.to(device)
            board_suit = board_suit.to(device)
            board_rank = board_rank.to(device)
            board_card_idx = board_card_idx.to(device)
            actions_occured = actions_occured.to(device)
            bet_sizes = bet_sizes.to(device)
            action = action.to(device)
            regret = regret.to(device)
            value = value.to(device)
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: Get the predicted values for the current states
            predicted_values = model(hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes, action)


            # Calculate the loss
            value = value.unsqueeze(1) 
            regret = regret.unsqueeze(1)
            output = torch.cat((regret, value), dim=1)
            # print("output", output)
            # print("predicted", predicted_values)
            loss = criterion(predicted_values.to(device), output.to(device))

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

for j in range(num_iterations):
    print("running round", j)
    #re-initialize model after simulating
    
    
    state = simulate(evaluation_function, num_rounds = num_rounds, training_round = j)
    evaluation_function = ValueNetwork()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(evaluation_function.parameters(), lr=0.0001)

    evaluation_function = nn.DataParallel(evaluation_function) #parallelize for GPU
    evaluation_function.to(device)
    train_loop(*state, evaluation_function, num_epochs=num_epochs, batch_size=batch_size)
    evaluation_function.eval()
    evaluation_function.to(eval_device) #evaluate on eval device (CPU)
    evaluation_function = evaluation_function.module #get the original model
    state = None #clear memory
    # explore_chance *= 0.95
    # if num_rounds < 100000:
    #     num_rounds *= 2

    #save model
    save_path = os.path.join(os.path.dirname(__file__), f"models/CFR-D_cp{j}.pth")
    torch.save(evaluation_function.state_dict(), save_path)
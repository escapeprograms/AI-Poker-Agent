from pypokerengine.api.game import setup_config, start_poker
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

device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")

def simulate(evaluation_function, num_rounds=3200):
    config = setup_config(max_round=num_rounds, initial_stack=10000, small_blind_amount=10)

    p1 = MinimaxPlayer(evaluation_function)
    p2 = MinimaxPlayer(evaluation_function)
    config.register_player(name="Eric", algorithm=p1)
    config.register_player(name="Evil", algorithm=p2)

    game_result = start_poker(config, verbose=1)
    #extract training data from both players
    hole_suit1, hole_rank1, hole_card_idx1, board_suit1, board_rank1, board_card_idx1, actions_occured1, bet_sizes1 = p1.train_embedded_state
    hole_suit2, hole_rank2, hole_card_idx2, board_suit2, board_rank2, board_card_idx2, actions_occured2, bet_sizes2 = p2.train_embedded_state
    values = p1.train_values + p2.train_values
    return hole_suit1 + hole_suit2, hole_rank1 + hole_rank2, hole_card_idx1 + hole_card_idx2,\
          board_suit1 + board_suit2, board_rank1 + board_rank2, board_card_idx1 + board_card_idx2,\
          actions_occured1 + actions_occured2, bet_sizes1 + bet_sizes2, values


########


# 2. Define Loss Function and Optimizer
evaluation_function = ValueNetwork()

criterion = nn.MSELoss()
optimizer = optim.Adam(evaluation_function.parameters(), lr=0.001)

num_epochs = 100
batch_size = 32
evaluation_function.to(device)

def train_loop(hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes, values, model):
    print(np.median(values), np.mean(values), np.std(values))
    dataset = ValueDataset(hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Set the model to training mode
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes, value) in enumerate(dataloader):

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: Get the predicted values for the current states
            predicted_values = model(hole_suit, hole_rank, hole_card_idx, board_suit, board_rank, board_card_idx, actions_occured, bet_sizes)


            # Calculate the loss
            loss = criterion(predicted_values, value)

            # Backward pass: Compute gradients
            loss.backward()

            # Optimization step: Update the model's weights
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0

        print(f"Epoch {epoch+1} finished")

    print("Finished Training")

#Run self-play to gather data, then train the value function
for j in range(5):
    state = simulate(evaluation_function, num_rounds = 3200)
    train_loop(*state, evaluation_function)
    evaluation_function.eval()

#save model
torch.save(evaluation_function.state_dict(), "models/evaluation_function.pth")
from pypokerengine.api.game import setup_config, start_poker
from minimax_player import MinimaxPlayer
from raise_player import RaisedPlayer
from randomplayer import RandomPlayer
from value_model import ValueNetwork
from training.value_dataset import ValueDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")

def simulate(evaluation_function, num_rounds=32000):
    config = setup_config(max_round=num_rounds, initial_stack=10000, small_blind_amount=10)

    p1 = MinimaxPlayer(evaluation_function)
    p2 = MinimaxPlayer(evaluation_function)
    config.register_player(name="Eric", algorithm=p1)
    config.register_player(name="Evil", algorithm=p2)

    game_result = start_poker(config, verbose=1)
    
    hole_cards = p1.train_hole_cards + p2.train_hole_cards
    round_states = p1.train_round_states + p2.train_round_states
    values = p1.train_values + p2.train_values
    return hole_cards, round_states, values


########


# 2. Define Loss Function and Optimizer
evaluation_function = ValueNetwork()

criterion = nn.MSELoss()
optimizer = optim.Adam(evaluation_function.parameters(), lr=0.001)

num_epochs = 1000
evaluation_function.to(device)

def train_loop(hole_cards, round_states, values, model):
    batch_size = 1
    dataset = ValueDataset(hole_cards, round_states, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for i, (hole_card, round_state, value) in enumerate(dataloader):

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: Get the predicted values for the current states
            predicted_values = model(hole_card, round_state)


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
for j in range(10):
    hole_cards, round_states, values = simulate(evaluation_function, num_rounds = 3)
    train_loop(hole_cards, round_states, values, evaluation_function)
    evaluation_function.eval()

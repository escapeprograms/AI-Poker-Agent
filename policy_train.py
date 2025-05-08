import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from policy import PolicyNet, testGame, model_wrapper
from utils import expectedValue
from datetime import datetime
from encode_state import encode_card

class PolicyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        print(data[0])
    #   "hand": hand1,
    #   "community": community[:CARDS_REVEALED[phase]],
    #   "pot": pot,
    #   "canRaise": canRaise,
    #   "action": action, # 0/1/2 for Fold/Call/Raise
    #   "value": EVs[action]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        goal = torch.tensor([0, 0, 0], dtype=torch.float32)
        goal[item["action"]] = 1

        cards = item["hand"]+item["community"]
        model.eval()
        suits, ranks, cardIds = [], [], []
        for card in cards:
            s, r, c = encode_card(card)
            suits.append(s)
            ranks.append(r)
            cardIds.append(c)
        return suits, ranks, cardIds, item["pot"], item["canRaise"], goal



def train_loop(data, model, num_epochs=30, batch_size=1, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.MSELoss()
    # print(np.median(values), np.mean(values), np.std(values))
    
    # model.to(device)
    dataset = PolicyDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Set the model to training mode
    model.train()
    print("# of training data:", len(dataset))
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (suits, ranks, cardIds, pot, canRaise, goal) in enumerate(dataloader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: Get the predicted values for the current states
            predicted_values = model(suits, ranks, cardIds, pot, canRaise)

            # Calculate the loss
            loss = criterion(predicted_values, goal[0])

            # Backward pass: Compute gradients
            loss.backward()

            # Optimization step: Update the model's weights
            optimizer.step()

            running_loss += loss.item()

        # Print statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {running_loss/len(dataset):.4f}")
        running_loss = 0.0

    print("Finished Training")

def train_iterations(model, num_iterations, num_tests=1, num_epochs=20, batch_size=1):
    #Run self-play to gather data, then train the value function
    for j in range(num_iterations):
        data = []
        for i in range(num_tests):
            testGame(model_wrapper(model), dataset=data)

        train_loop(data, model, num_epochs=num_epochs, batch_size=batch_size)
        model.eval()

        #save model
        if j % 10 == 9:
            torch.save(model.state_dict(), f"models/policy[{datetime.today().strftime('%d-%m-%y %H.%M.%S')}].pth")

if __name__ == "__main__":
    model = PolicyNet()
    model.load_state_dict(torch.load("models/evaluation_function[latest-test].pth"))

    train_iterations(model, 100, num_tests=2, num_epochs=2, batch_size=1)
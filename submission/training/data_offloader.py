import torch
import numpy as np
import os

storage_path = "submission/training/training_segments"

def store_training_data(training_data, segment_ind):
    #store training data
    data_types = ["hole_suit", "hole_rank", "hole_card_idx", 
                  "board_suit", "board_rank", "board_card_idx", 
                  "actions_occured", "bet_sizes", "action_num", "regret", "value"]
    
    for i, data in enumerate(training_data):
        np.save(f"{storage_path}/{data_types[i]}_{segment_ind}.pt", data.numpy())


def load_training_data(num_segments):
    #load training data
    data_types = ["hole_suit", "hole_rank", "hole_card_idx", 
                  "board_suit", "board_rank", "board_card_idx", 
                  "actions_occured", "bet_sizes", "action_num", "regret", "value"]
    
    training_data = []
    for i in range(len(data_types)):
        training_data.append([])
        for segment_ind in range(int(num_segments)):
            data = np.load(f"{storage_path}/{data_types[i]}_{segment_ind}.pt")
            data = torch.from_numpy(data)
            os.remove(f"{storage_path}/{data_types[i]}_{segment_ind}.pt")
            training_data[i] += data

    return training_data

import torch
import torch.nn as nn 
from main import BOARD_WIDTH, BOARD_HEIGHT, initialize_board, display_board_with_labels, place_dandelion, spread_seeds, check_dandelion_win, convert_user_input, directions, direction_names, validate_row_input, validate_col_input, validate_direction_input, play_game

LEARNING_RATE = 0.001

INPUT_SIZE = BOARD_WIDTH * BOARD_HEIGHT * 2
HIDDEN_SIZE = 100
OUTPUT_SIZE = 8

def board_to_tensor(board):
    dandelion_tensor = torch.tensor([[1 if cell == 1 else 0 for cell in row] for row in board]).view(-1).float()
    seed_tensor = torch.tensor([[1 if cell == 2 else 0 for cell in row] for row in board]).view(-1).float()
    return torch.cat((dandelion_tensor, seed_tensor))


class WindNeuralNetwork(nn.Module):
    # might add more layers... Just get prototype working first

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
        )
        '''
        layers = OrderedDict()
        for i in range(len(NN_SIZE) - 1):
            layers[f"layer_{i}"] = nn.Linear(NN_SIZE[i], NN_SIZE[i+1])
            if i < len(NN_SIZE) - 2:  # No ReLU after last layer
                layers[f"relu_{i}"] = nn.ReLU()
        self.linear_relu_stack = nn.Sequential(layers)
        '''

    def forward(self, x):
        logits = self.linear_relu_stack(x)        
        return logits

def train_nn():
    wind_brain  = WindNeuralNetwork()
    optimizer = torch.optim.AdamW(wind_brain.parameters(), lr=LEARNING_RATE)

    board = [[2, 1, 2, 1, 2], [0, 0, 2, 2, 2], [2, 2, 1, 2, 2], [2, 2, 2, 2, 1], [1, 0, 2, 2, 2]]
    # to do: make a bunch of boards and train on them
    T = board_to_tensor(board)

    logits = wind_brain(T)  # that calls forward because __call__ is coded magic backend
    print("logits" , logits)

if __name__ == "__main__":
    print("windBrain.py")
    train_nn()
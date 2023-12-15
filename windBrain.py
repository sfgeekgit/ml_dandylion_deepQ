import torch
import torch.nn as nn 
from game import BOARD_WIDTH, BOARD_HEIGHT, initialize_board, display_board_with_labels, place_dandelion, spread_seeds, check_dandelion_win, convert_user_input, directions, direction_names, validate_row_input, validate_col_input, validate_direction_input, play_game

LEARNING_RATE = 0.001


NUM_DIR = 8 # will always be 8, but here for clarity
INPUT_SIZE = NUM_DIR + BOARD_WIDTH * BOARD_HEIGHT * 2
HIDDEN_SIZE = 100
OUTPUT_SIZE = 8

def board_to_tensor(board, available_directions):
    # 1st 8 cells are available directions
    # next 25 cells are dandelions
    # next 25 cells are seeds
    direction_tensor = torch.tensor([1 if direction else 0 for direction in available_directions]).float()
    dandelion_tensor  = torch.tensor([[1 if cell == 1 else 0 for cell in row] for row in board]).view(-1).float()
    seed_tensor       = torch.tensor([[1 if cell == 2 else 0 for cell in row] for row in board]).view(-1).float()
    
    print(f"{direction_tensor=}")

    return torch.cat((direction_tensor, dandelion_tensor, seed_tensor))


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

def reward_func(board, direction):
    print("reward_func")
    print(f"{board=}")
    rews = {"win": 100, 
            "lose": -100, 
            "illegal": -100, 
            "meh": 0
            }
    print(f"{board=}")
    print(f"{direction=}")
    print(f"{rews=}")
    quit()

    reward = rews["meh"]
    if not direction in direction_names:
        reward = rews["illegal"]
        return reward
    
    new_board = spread_seeds(board, direction)
    wind_lost = check_dandelion_win(new_board)
    if wind_lost:
        reward = rews["lose"]
    
    return reward

def reward_func_by_AI (board, direction):
    # um... I just wrote the function name, litterally all this code showed up..
    # other than this comment... I was going to just give all for win or illegale move..
    # ai comment and code follows:
    # reward is the number of seeds that land on dandelions
    # so we need to spread the seeds and then count the number of dandelions
    new_board = spread_seeds(board, direction)
    reward = 0
    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            if new_board[row][col] == 2:
                reward += 1
    return reward


def train_nn():
    wind_brain  = WindNeuralNetwork()
    optimizer = torch.optim.AdamW(wind_brain.parameters(), lr=LEARNING_RATE)

    available_directions = [0,1,0,1,0,1,0,1]     

    board = [[2, 1, 2, 1, 2], [0, 0, 2, 2, 2], [2, 2, 1, 2, 2], [2, 2, 2, 2, 1], [1, 0, 2, 2, 2]]
    # to do: make a bunch of boards and train on them
    T = board_to_tensor(board, available_directions)
    print("T", T)


    logits = wind_brain(T)  # that calls forward because __call__ is coded magic backend
    print("logits" , logits)

    #to_from_logs = logits.argmax()  # do this after training
    #wind_direction = logits.argmax()  # do this after training

    


if __name__ == "__main__":
    print("windBrain.py")
    train_nn()
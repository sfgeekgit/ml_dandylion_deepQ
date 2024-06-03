import torch
import torch.nn as nn 
import torch.nn.functional as F
import random
from game import BOARD_WIDTH, BOARD_HEIGHT, initialize_board, display_board_with_labels, place_dandelion, spread_seeds, check_dandelion_win, convert_user_input, dir_pairs, direction_names, validate_row_input, validate_col_input, validate_direction_input, play_game



LEARNING_RATE = 0.001

DECAY=0.9

NUM_DIR = 8 # will always be 8, but here for clarity
INPUT_SIZE = NUM_DIR + BOARD_WIDTH * BOARD_HEIGHT * 2
HIDDEN_SIZE = 100
OUTPUT_SIZE = NUM_DIR

EPOCHS = 2000


def board_state_to_tensor(board, available_directions):
    # 1st 8 cells are available directions
    # next 25 cells are dandelions
    # next 25 cells are seeds
    direction_tensor = torch.tensor([1 if direction else 0 for direction in available_directions]).float()
    dandelion_tensor  = torch.tensor([[1 if cell == 1 else 0 for cell in row] for row in board]).view(-1).float()
    seed_tensor       = torch.tensor([[1 if cell == 2 else 0 for cell in row] for row in board]).view(-1).float()
    return torch.cat((direction_tensor, dandelion_tensor, seed_tensor))

def board_state_from_tensor(tensor):
    # 1st 8 cells are available directions
    # next 25 cells are dandelions
    # next 25 cells are seeds
    direction_list = [1 if direction else 0 for direction in tensor[:NUM_DIR]]
    grid_size = BOARD_WIDTH * BOARD_HEIGHT
    dandelion_list  = tensor[NUM_DIR:NUM_DIR+grid_size].tolist()
    seed_list       = tensor[NUM_DIR+grid_size:].tolist()
    board = []
    for row in range(BOARD_HEIGHT):
        board.append([])
        for col in range(BOARD_WIDTH):
            board[row].append(0)
    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            if dandelion_list[row * BOARD_WIDTH + col] == 1.:
                board[row][col] = 1
            elif seed_list[row * BOARD_WIDTH + col] == 1.:
                board[row][col] = 2    
    #out = [direction_list, board]
    #print(f"Board State From Tensor (should be directions, board) {out=}")
    return [direction_list, board]



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

def reward_func(boardTensor, direction):

    rewards = {"win": 100, 
            "lose": -100, 
            "illegal": -100, 
            "meh": 0
            }

    board_state = board_state_from_tensor(boardTensor)
    avail_dirs = board_state[0]
    if not avail_dirs[direction]:
        #print("illegal move")
        return rewards["illegal"]
    old_board = board_state[1]
    #print(f"{old_board=}")

    dir_tuple = dir_pairs[direction]
    new_board = spread_seeds(old_board, dir_tuple)
    #print(f"{new_board=}")

    wind_lost = check_dandelion_win(new_board)
    if wind_lost:
        #print("Wind lost")
        return rewards["lose"]
    else:         # havn't lost yet!
        # if this func was called with only 2 avail directions, then wind has won
        # (this will be the wind's 7th move without losing)
        moves_left  = sum(avail_dirs)
        if moves_left <= 2:
            return rewards["win"]
    return rewards["meh"]


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


    for stepnum in range(EPOCHS):


        #print()
        #print()

        available_directions = [0,1,0,1,0,1,0,1]
        #print("available_directions", available_directions)


        board = [[2, 1, 2, 1, 2], [0, 0, 2, 2, 2], [2, 2, 1, 2, 2], [2, 2, 2, 2, 1], [1, 0, 2, 2, 2]]
        #print("board", board)
        # to do: make a bunch of boards and train on them
        boardStateTensor = board_state_to_tensor(board, available_directions)
        #print("boardStateTensor", boardStateTensor)

        #bb = board_state_from_tensor(boardStateTensor)
        #print("bb", bb)
        


        logits = wind_brain(boardStateTensor)  # that calls forward because __call__ is coded magic backend
        #print("logits" , logits)


        # now pick one random logit to check
        # TODO: why just check one??? Maybe try "exhaustive" and check all? (as a later version)
        # but meanwhile...
        # use that one logit as the bellman left.
        # then (I think) make that move and (I think) get the max logit as bellman_rigt


        # Now pick one random direction and use it's logit as bellman left, then check that same pick as bellman right
        rand_dir_check = random.randint(0, NUM_DIR-1)


        checker_one_hots = F.one_hot(torch.tensor(rand_dir_check), NUM_DIR)
        logits = logits * checker_one_hots

        #print(f"{rand_dir_check=}")
        #print(f"{checker_one_hots=}")
        #print(f"{logits=}")
        
        bellman_left = logits.sum()  # woot!
        #print(f"{bellman_left=}")
        #quit()

        # OK, now have bellman_left. Make that move and get the max reward as bellman right?
        reward = reward_func(boardStateTensor, rand_dir_check)
        #print(f"{reward=}")


        if reward == 0:
            keep_playing = torch.tensor(1)   # keep playing
        else:
            keep_playing = torch.tensor(0)   # terminal state, zero out the right logits, only use the reward
        
        #print(f"{     board=}")
        next_state = spread_seeds(board, dir_pairs[rand_dir_check])
        #print(f"{next_state=}")
        # Remove choice from available directions
        next_available_directions = available_directions.copy()
        next_available_directions[rand_dir_check] = 0
        right_input = board_state_to_tensor(next_state, next_available_directions)

        right_logits = wind_brain(right_input)

        max_right_logits = right_logits.max(dim=0).values

        bellman_right = reward + keep_playing * DECAY * max_right_logits

        # MSE
        loss = F.mse_loss(bellman_left, bellman_right)

        if stepnum % 10 == 0:
            print(f"loss {loss.item()}")


        #### Now do the optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()







if __name__ == "__main__":
    print("windBrain.py")
    train_nn()
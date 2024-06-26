import torch
import torch.nn as nn 
import torch.nn.functional as F
import random
from game import BOARD_WIDTH, BOARD_HEIGHT, initialize_board, display_board_with_labels, place_dandelion, spread_seeds, check_dandelion_win, convert_user_input, dir_pairs, direction_names, validate_row_input, validate_col_input, validate_direction_input, play_game

BOARD_WIDTH = 0 # DEV!
BOARD_HEIGHT = 0 # DEV!


LEARNING_RATE = 0.005

DECAY=0.98

NUM_DIR = 8 # will always be 8, but here for clarity
INPUT_SIZE = NUM_DIR + BOARD_WIDTH * BOARD_HEIGHT * 2
#HIDDEN_SIZE = 100
HIDDEN_SIZE = INPUT_SIZE # *2
OUTPUT_SIZE = NUM_DIR

EPOCHS = 10000


reward_vals = {
    "win": 1000, 
    "lose": -100, 
    "illegal": -100, 
    #"meh": 0
    "meh": 10 
}
                

def board_state_to_tensor(available_directions, board):
    # 1st 8 cells are available directions
    # next 25 cells are dandelions
    # next 25 cells are seeds
    direction_tensor = torch.tensor([1 if direction else 0 for direction in available_directions]).float()
    if BOARD_HEIGHT == 0:  # dev
        return direction_tensor 
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

    board_state = board_state_from_tensor(boardTensor)
    avail_dirs = board_state[0]

    #print(f"{avail_dirs=} {direction=}")
    if not avail_dirs[direction]:
        #print("illegal move")
        #return rewards["illegal"]
        return "illegal"
    if BOARD_WIDTH > 0:
        old_board = board_state[1]
        #print(f"{old_board=}")
        dir_tuple = dir_pairs[direction]
        new_board = spread_seeds(old_board, dir_tuple)
    else:
        old_board = []
        new_board = []
    #print(f"{new_board=}")

    wind_lost = check_dandelion_win(new_board) and BOARD_WIDTH > 0
    if wind_lost:
        #print("Wind lost")
        #quit()
        #return rewards["lose"]
        return "lose"
    else:         # havn't lost yet!
        # if this func was called with only 2 avail directions, then wind has won
        # (this will be the wind's 7th move without losing)
        moves_left  = sum(avail_dirs)
        if moves_left <= 2:
            #return rewards["win"]
            #print("Wind won")
            return "win"
    # return rewards["meh"]
    #print("meh")
    return "meh"



def rand_training_board() -> tuple[list[list[int]], list[int]]:
    available_directions = [1,1,1,0,0,0,0,1]

    #num_ones = random.randint(1, 6)
    #num_zeros = NUM_DIR - num_ones
    #available_directions = [1] * num_ones + [0] * num_zeros
    random.shuffle(available_directions)

    board = [[2, 1, 2, 1, 2], [0, 0, 2, 2, 2], [2, 2, 1, 2, 2], [2, 2, 2, 2, 1], [1, 0, 2, 2, 2]]
    #available_directions = [random.randint(0, 1) for _ in range(NUM_DIR)]
    return board, available_directions




def train_nn():
    wind_brain  = WindNeuralNetwork()
    optimizer = torch.optim.AdamW(wind_brain.parameters(), lr=LEARNING_RATE)


    for stepnum in range(EPOCHS):


        #print()
        #print()




        #board = [[2, 1, 2, 1, 2], [0, 0, 2, 2, 2], [2, 2, 1, 2, 2], [2, 2, 2, 2, 1], [1, 0, 2, 2, 2]]
        #print("board", board)
        # to do: make a bunch of boards and train on them

        board, available_directions = rand_training_board()
        boardStateTensor = board_state_to_tensor(available_directions, board)


        #bb = board_state_from_tensor(boardStateTensor)
        #print("bb", bb)
        


        logits = wind_brain(boardStateTensor)  # that calls forward because __call__ is coded magic backend
        #print("logits" , logits)


        # TRY EXHAUSTIVE!!
        # TRY EXHAUSTIVE!!
        # TRY EXHAUSTIVE!!
        # TRY EXHAUSTIVE!!
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
        reward_type = reward_func(boardStateTensor, rand_dir_check)
        right_reward = reward_vals[reward_type]
        #print(f"{reward=}")


        if reward_type == "meh": # keep playing
            keep_playing = torch.tensor(1)   # keep playing
        else:
            keep_playing = torch.tensor(0)   # terminal state, zero out the right logits, only use the reward

        #print(f"{reward_type=} {keep_playing=}  {reward=}")


        '''
        if reward >= 20 or reward <= -20:
            keep_playing = torch.tensor(1)   # keep playing
        else:
            keep_playing = torch.tensor(0)   # terminal state, zero out the right logits, only use the reward
        '''
        #print(f"{     board=}")
        next_state = spread_seeds(board, dir_pairs[rand_dir_check])
        #print(f"{next_state=}")
        # Remove choice from available directions
        next_available_directions = available_directions.copy()
        next_available_directions[rand_dir_check] = 0
        right_input = board_state_to_tensor(next_available_directions, next_state)

        right_logits = wind_brain(right_input)

        max_right_logits = right_logits.max(dim=0).values

        bellman_right = right_reward + keep_playing * DECAY * max_right_logits

        # MSE
        loss = F.mse_loss(bellman_left, bellman_right)

        if stepnum % 50 == 0:
            print(f"loss {loss.item()}")
            print(f"{right_logits=}")
            print(f"{max_right_logits=}")

            print(f"{rand_dir_check=}")
            print(f"{checker_one_hots=}")
            print(f"{logits=}")
            
            print(f"{bellman_left=}")


            print(f"{right_reward=}\n\n\n")

        if stepnum % 1000 == 0:
            print(f"stepnum {stepnum} of {EPOCHS}")
            test_wind_brain(wind_brain)

        if stepnum == 4000:
            print(f"{logits=} \n{right_logits=} \n{bellman_left=} \n{bellman_right=} \n{right_reward=} \n{keep_playing=} \n{max_right_logits=}")
            quit()

        #### Now do the optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return wind_brain



def test_wind_brain(wind_brain):
    print("\n\n Test Wind Brain")
    board, available_directions = rand_training_board() 
    boardStateTensor = board_state_to_tensor(available_directions, board)
    logits = wind_brain(boardStateTensor)  # that calls forward because __call__ is coded magic backend
    print("avails" , available_directions)
    print("test logits" , logits)




if __name__ == "__main__":
    wind_brain = train_nn()
    test_wind_brain(wind_brain)




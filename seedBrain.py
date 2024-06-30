import torch
import torch.nn as nn 
import torch.nn.functional as F
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

from game import BOARD_WIDTH, BOARD_HEIGHT, NUM_DIR, initialize_board, dir_pairs, place_dandelion, spread_seeds, check_dandelion_win #, convert_user_input, direction_names, validate_row_input, validate_col_input, validate_direction_input, play_game,  display_board_with_labels
from brainlib import board_state_to_tensor, board_state_from_tensor


LEARNING_RATE = 0.002

EXPLORATION_PROB = 0.01

INPUT_SIZE = NUM_DIR + 2 *(BOARD_HEIGHT * BOARD_WIDTH)
HIDDEN_SIZE = INPUT_SIZE # * 2
OUTPUT_SIZE = BOARD_HEIGHT * BOARD_WIDTH

EPOCHS = 500

# Gamma aka discount factor for future rewards or "Decay"
GAMMA = 0.97  


reward_vals = {
    "win": 100, 
    "illegal": -100, 
    "lose": -50,
    "meh": 15 
}


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        # just to start. Will probably add at least one more layer


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

seedbrain = DQN()
optimizer = torch.optim.Adam(seedbrain.parameters(), lr=LEARNING_RATE)

def game_step(board_state_tensor, action):
    #new_state, result, done = game_step(board_state_tensor, action)
    dir_list, board_grid = board_state_from_tensor(board_state_tensor)

    place_row = action//BOARD_WIDTH
    place_col = action%BOARD_WIDTH


    if board_grid[place_row][place_col] == 1:
        done = 1
        return board_state_tensor, 'illegal', done

    board_grid[place_row][place_col] = 1    


    won = check_dandelion_win(board_grid)
    if won:
        done = 1
        return board_state_tensor, 'win', done

    # todo, wind moves, and check wind win
    lost = False
    if lost:
        done = 1
        return board_state_tensor, 'lose', done

    # still here? Meh.
    new_state = board_state_to_tensor(dir_list, board_grid)
    result = 'meh'
    done = 0

    return new_state, result, done  # Continue game


def wind_move():
    pass


def train_seeds():
    for epoch in range(EPOCHS):            
        # Initialize game state in tensor
        used_dirs = [0] * NUM_DIR  
        board_grid = initialize_board()
        board_state_tensor = board_state_to_tensor(used_dirs, board_grid)

        mv_cnt = 0  # just for dev tracking

        # each epoch, play until a terminal state is reached
        done = False
        while not done:
            # generate prediction
            q_values_pred = seedbrain(board_state_tensor)  # calls forward()
            q_values_pred = q_values_pred.unsqueeze(0)  # get correct tensor shape 

            ### init target q values
            ##target_q_values = q_values_pred.clone()  # just make with zeros() for exhaustive
            target_q_values = torch.zeros(1, OUTPUT_SIZE)
            
            for action in range(OUTPUT_SIZE):
                next_state, result, done = game_step(board_state_tensor, action)
                reward_val = reward_vals[result]

                with torch.no_grad():
                    next_q_values = seedbrain(next_state)
                max_next_q_value = torch.max(next_q_values).item()

                if done:
                    bellman_right = reward_val
                else:
                    bellman_right = reward_val + GAMMA * max_next_q_value  

                target_q_values[0, action] = bellman_right  

            # all actions checked
            # now we have bellman_right for every action stored in target_q_values


            # Learning step
            # this should happen per move (not per epoch)
            # todo, maybe this could be batched???? Batch and run once per epoch? (on GPU)
            optimizer.zero_grad()
            loss = F.mse_loss(q_values_pred, target_q_values) 
            loss.backward()
            optimizer.step()


            # get action for next step
            if random.random() < EXPLORATION_PROB:  # explore
                action = random.randint(0, OUTPUT_SIZE-1)
            else:                                   # exploit best guess
                action = seedbrain(board_state_tensor).argmax().item()

            # take action, set board for next step
            next_state, result, done = game_step(board_state_tensor, action)
            board_state_tensor = next_state

            mv_cnt += 1  # just for dev accounting

        if epoch % 50 == 0:
            print(f"{epoch=} {mv_cnt=} {result=} {loss.item()=}")
        mv_cnt = 0


if __name__ == "__main__":
    train_seeds()


import torch
import torch.nn as nn 
import torch.nn.functional as F
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

from game import BOARD_WIDTH, BOARD_HEIGHT, NUM_DIR, initialize_board, dir_pairs, place_dandelion, spread_seeds, check_dandelion_win #, convert_user_input, direction_names, validate_row_input, validate_col_input, validate_direction_input, play_game,  display_board_with_labels
from brainlib import board_state_to_tensor, board_state_from_tensor


LEARNING_RATE = 0.002

EXPLORATION_PROB = 0.01
EXPLORATION_PROB_STEPS = {20:0.2,   # first x percent of epochs -> y
                          30:0.001,  # after x percent, etc
                          50:0}     

INPUT_SIZE = NUM_DIR + 2 *(BOARD_HEIGHT * BOARD_WIDTH)
HIDDEN_SIZE = INPUT_SIZE * 2
OUTPUT_SIZE = BOARD_HEIGHT * BOARD_WIDTH

EPOCHS = 15001

# Gamma aka discount factor for future rewards or "Decay"
GAMMA = 0.99  


reward_vals = {
    "win": 100, 
    "illegal": -100, 
    "lose": -70,
    "meh": 5 
}


#BATCH_SIZE = 4 # Dev. Probably make bigger like 32 or more

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        # just to start. Will probably add at least one more layer
        self.to(device)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def game_step(board_state_tensor, action, wind_action = None):
    board_state_tensor = board_state_tensor.to(device)
    #new_state, result, done = game_step(board_state_tensor, action)
    dir_list, board_grid = board_state_from_tensor(board_state_tensor)

    place_row = action//BOARD_WIDTH
    place_col = action%BOARD_WIDTH


    if board_grid[place_row][place_col] == 1:
        done = 1
        return board_state_tensor, 'illegal', done

    board_grid[place_row][place_col] = 1    

    # wind blows. Check dandelion win. If not win, check wind win.
    dir_list, board_grid = wind_move(dir_list, board_grid, wind_action)

 
    won = check_dandelion_win(board_grid)
    if won:
        done = 1
        return board_state_tensor, 'win', done

    # still here? Check wind win
    if sum(dir_list) >= 7:  # wind won! Lost!
        done = 1
        return board_state_tensor, 'lose', done


    # still here? Meh.
    new_state = board_state_to_tensor(dir_list, board_grid, device)
    result = 'meh'
    done = 0

    return new_state, result, done  # Continue game


def wind_move(direction_list, board_grid, new_dir = None):
    # optional: pass the wind_action to the function, so can be deterministic in exhuastive search

    if new_dir is None:
        # select a random direction that has NOT already been used:
        #if sum(direction_list) >= NUM_DIR: # sanity check
        #    print("Error, too many wind moves")
        #    quit()
        
        new_dir = random.randint(0, NUM_DIR-1)
        while direction_list[new_dir] == 1:
            new_dir = random.randint(0, NUM_DIR-1)

    direction_list[new_dir] = 1
    dir_tuple = dir_pairs[new_dir]
    board_grid = spread_seeds(board_grid, dir_tuple)

    return direction_list, board_grid


def train_seeds():

    w_cnt = l_cnt =  0

    for epoch in range(EPOCHS):

        #used_dirs_batch = torch.zeros((BATCH_SIZE, NUM_DIR), dtype=torch.float32, device=device)
        #board_grid_batch = torch.zeros((BATCH_SIZE, BOARD_HEIGHT * BOARD_WIDTH), dtype=torch.float32, device=device)



        # Initialize game state in tensor
        used_dirs = [0] * NUM_DIR  
        board_grid = initialize_board()
        board_state_tensor = board_state_to_tensor(used_dirs, board_grid, device)  # Pass device here

        mv_cnt = 0  # just for dev tracking

        EXPLORATION_PROB = 0
        run_percent = (epoch + 1) / EPOCHS * 100
        for percent, prob in EXPLORATION_PROB_STEPS.items():
            if run_percent < percent:
                EXPLORATION_PROB = prob
                break

        # each epoch, play until a terminal state is reached
        done = False
        while not done:
            # generate prediction
            q_values_pred = seedbrain(board_state_tensor)  # calls forward()
            q_values_pred = q_values_pred.unsqueeze(0)  # get correct tensor shape 
            # to maybe do: Explore batches


            ### init target q values
            ##target_q_values = q_values_pred.clone()  # just make with zeros() for exhaustive
            target_q_values = torch.zeros(1, OUTPUT_SIZE).to(device)
            
            # choose one wind action and apply to all actions in exhaustive search
            # This is good for version 1, where opponent is random anyway
            # in real play, opp move might depend on seed's action
            direction_list, board_grid = board_state_from_tensor(board_state_tensor)
            wind_action = random.randint(0, NUM_DIR-1)
            while direction_list[wind_action] == 1:
                wind_action = random.randint(0, NUM_DIR-1)


            #exhaustive search of seed options, using one wind action
            # perhaps maaaaybe should pick "best of worst" do exhaustive of exhaustive. For each seed action, check all wind options, and take the worst (for seed)
            # ^^ that would be vs an "ideal" opponent, instead of random ("ideal" with only one move look ahead)
            for action in range(OUTPUT_SIZE):
                next_state, result, done = game_step(board_state_tensor, action, wind_action)
                reward_val = reward_vals[result]

                with torch.no_grad():
                    next_q_values = seedbrain(next_state)  # calls forward
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
                action = seedbrain(board_state_tensor).argmax().item()  # calls forward

            # take action, set board for next step
            next_state, result, done = game_step(board_state_tensor, action)
            board_state_tensor = next_state

            mv_cnt += 1  # just for dev accounting

        if result == "win":
            #print("W", end="")
            w_cnt += 1
        if result == "lose":    
            #print("L", end="")
            l_cnt += 1

        e_mod = 400
        if epoch % e_mod == 0:
            wperc = int(w_cnt / e_mod * 100)
            print ("W" * wperc)
            print(f"{epoch=} r%{round(run_percent, 1)} {wperc=} {w_cnt=} {l_cnt=} w+l={w_cnt+l_cnt}= EXP {EXPLORATION_PROB} recloss {round(loss.item(), 4)}")
            w_cnt = l_cnt = 0
        mv_cnt = 0


if __name__ == "__main__":

    seedbrain = DQN().to(device)  
    optimizer = torch.optim.Adam(seedbrain.parameters(), lr=LEARNING_RATE)
    train_seeds()

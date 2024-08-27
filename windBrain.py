'''
Current status:
It works!
This is about where I want v1 to be :)
After a few thousand training runs (with exploration prob = 0.2), it never makes an illegal move (unless forced to explore)
Plays vs an opponent that randomly places seeds. 
after 20,000 training epochs, it consistantly wins over 95% of the time (vs a dumb random opponent)
'''

import torch
import torch.nn as nn 
import torch.nn.functional as F
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from game import * # BOARD_WIDTH, BOARD_HEIGHT, NUM_DIR, initialize_board, dir_pairs, place_dandelion, spread_seeds, check_dandelion_win  #, convert_user_input, direction_names, validate_row_input, validate_col_input, validate_direction_input, play_game,  display_board_with_labels
from brainlib import *


LEARNING_RATE = 0.002
# learning rate is steped down in the code with:  scheduler = ReduceLROnPlateau...

#EXPLORATION_PROB = 0.01
EXPLORATION_PROB_STEPS = {15:0.2,    # first x percent of epochs -> y
                          30:0.001,  # until x percent, etc
                          50:0}     

#NUM_DIR = 8 # will always be 8, but here for clarity


INPUT_SIZE = NUM_DIR + 2 *(BOARD_HEIGHT * BOARD_WIDTH)
HIDDEN_SIZE = INPUT_SIZE *2
MIDDLE_LAYERS = [HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE]
OUTPUT_SIZE = NUM_DIR

EPOCHS = 240000
EPOCHS = 150000



# Gamma aka discount factor for future rewards or "Decay"
GAMMA = 0.99  

reward_vals = {
    "win": 100, 
    "illegal": -100, 
    "lose": -75,
    "meh_base": 20,
    "spread_pen": -1 # per dot on the board (seed spread)
}



def game_step(state, action):
    used_dir_list, board_grid = board_state_from_tensor(state)

    # check if action is legal
    if used_dir_list[action] == 1:
        return state, reward_vals['illegal'], 1  # Illegal move
 
    new_used_dir_list = used_dir_list[:]
    new_used_dir_list[action] = 1

    # spread seeds
    dir_tuple = dir_pairs[action]
    board_grid = spread_seeds(board_grid, dir_tuple)


    # check if dandelion wins
    dandelion_win = check_dandelion_win(board_grid)
    if dandelion_win:
        new_state = board_state_to_tensor(new_used_dir_list, board_grid)
        return new_state, reward_vals['lose'], 1  # Game lost

    # still here? Did I win?
    if sum(new_used_dir_list) >= 7:  # wind has gone 7 times. Winner!
        new_state = board_state_to_tensor(new_used_dir_list, board_grid)
        return new_state, reward_vals['win'], 1  # Game won

    # ddlion move
    new_state = ddlion_move(new_used_dir_list, board_grid)

    # check again if dandelion wins (After ddlion move)
    dandelion_win = check_dandelion_win(board_grid)
    if dandelion_win:
        return new_state, reward_vals['lose'], 1  # Game lost

    # still here? Keep playing
    # small penalty for seed spread
    seed_count = sum(1 for row in board_grid for value in row if value == 2)
    seed_pen = -seed_count * reward_vals['spread_pen']
    if seed_pen >= reward_vals['meh_base']:
        reward = 1
    else:
        reward = reward_vals['meh_base'] - seed_pen

    #return new_state, reward_vals['meh_base'], 0  # Continue game
    return new_state, reward, 0  # Continue game


def ddlion_move(used_dir_list, board_grid):
    # make a random move for now
    x=random.randint(0, BOARD_HEIGHT-1)
    y=random.randint(0, BOARD_WIDTH-1)
    board_grid[x][y] = 1
    new_state = board_state_to_tensor(used_dir_list, board_grid)
    return new_state


############### 
# begin
############### 
def train_wind():
    rec_loss_mod, rec_loss_idx = 40, 0
    rec_loss = [0] * rec_loss_mod
    wcnt = lcnt =0


    for epoch in range(EPOCHS):
        # Initialize game state
        used_dirs = [0] * NUM_DIR  
        board_grid = initialize_board()
        
        # seeds make first move
        state_tensor = ddlion_move(used_dirs, board_grid)


        run_percent = (epoch + 1) / EPOCHS * 100
        for percent, prob in EXPLORATION_PROB_STEPS.items():
            if run_percent < percent:
                EXPLORATION_PROB = prob
                break

        done = False

        # each epoch, play until a terminal state is reached
        while not done:
            # Generate predictions
            q_values_pred = model(state_tensor)   # calls forward
            q_values_pred = q_values_pred.unsqueeze(0)  # get correct tensor shape

            
            # init target_q_values with zeroes
            target_q_values = torch.zeros(1, NUM_DIR)

            # explore all actions (exhaustive checking for given current board state)
            # this makes the "loss" numbers look bigger, but is much more efficient overall
            for action in range(NUM_DIR):
                next_state, reward, done = game_step(state_tensor, action)

                with torch.no_grad():     # don't set the gradients becase we will not call backward on this
                    next_q_values = model(next_state)  # calls forward()

                max_next_q_value = torch.max(next_q_values).item()
                if done:
                    bellman_right = reward
                else:
                    bellman_right = reward + GAMMA * max_next_q_value  # This is the target value

                target_q_values[0, action] = bellman_right  # Update using Bellman equation
            # all actions checked
            # now we have bellman_right for every action stored in target_q_values



            # Learning step
            # this should happen per move (not per epoch)
            optimizer.zero_grad()
            loss = F.mse_loss(q_values_pred, target_q_values) 
            loss.backward()
            optimizer.step()



            # select and make next move
            if random.random() < EXPLORATION_PROB:      # explore? 
                #expl = 'explore'
                action = random.randint(0, NUM_DIR-1)
            else:                                       # else exploit
                #expl = 'exploit'
                action = torch.argmax(q_values_pred).item()

            next_state, reward, done = game_step(state_tensor, action)
            state_tensor = next_state


            if reward == reward_vals['win']:
                wcnt += 1
                #print("W", end="")
            if reward == reward_vals['lose']:
                lcnt += 1
                #print("L", end="")


        #print(f"^^^^^^^^^^^         {epoch=}        RUN DONE   ^^^^^^^^^^^^^^^^^^^^^\n\n")

        
        rec_loss[rec_loss_idx] = loss.item()
        rec_loss_idx += 1
        if rec_loss_idx % rec_loss_mod == 0:
            rec_loss_idx = 0


        step_mod = 50
        if epoch % step_mod == 0:
            # update the Learning Rate if the loss has platued
            if epoch > 2* rec_loss_mod and run_percent > 50 and loss.item() < 100:
                avg_loss = sum(rec_loss) / rec_loss_mod
                if avg_loss < 100:
                    scheduler.step(avg_loss)


        epoch_mod = 400
        if (epoch + 0) % epoch_mod == 0:
            learn_rate = optimizer.param_groups[0]['lr']
            wpct = round(100*wcnt / epoch_mod,1)
            lpct = round(100*lcnt / epoch_mod,1)
            finis = wcnt + lcnt
            print(f"\nEpoch {epoch}, run_perc {round(run_percent, 1)} {wpct=} {lpct=}  {finis=} {EXPLORATION_PROB=} {learn_rate=}") #\n{q_values_pred=}")
            #print(f"  {q_values_pred=}\n{target_q_values=}\n")
            print(f"recent loss avg: -----  {sum(rec_loss)/rec_loss_mod}")
            #print()


            '''
            if wcnt > 99:
                good_pass_cnt += 1
            if good_pass_cnt > 5:
                print("got good by epoch", epoch +1)
                #quit()
            '''
            wcnt = lcnt =  0

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")    

    #seedbrain = DQN(INPUT_SIZE, MIDDLE_LAYERS, OUTPUT_SIZE, device)
    #optimizer = torch.optim.Adam(seedbrain.parameters(), lr=LEARNING_RATE)
    #train_wind()


    #model = DQN()
    model = DQN(INPUT_SIZE, MIDDLE_LAYERS, OUTPUT_SIZE, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=EPOCHS//100, cooldown=EPOCHS//10)
    train_wind()

    # Save the model
    model_subdir, subdir_num = get_next_model_subdir("models/wind")




    #os.makedirs(model_subdir, exist_ok=True)
    #model_save_path = os.path.join(model_subdir, "windbrain.pth")
    #torch.save(model.state_dict(), model_save_path)
    #print(f"Model saved to {model_save_path}")

    # Save the parameters
    params = {
        "EPOCHS": EPOCHS,
        "GAMMA": GAMMA,
        "LEARNING_RATE": LEARNING_RATE,
        "EXPLORATION_PROB_STEPS": EXPLORATION_PROB_STEPS,
        "reward_vals": reward_vals,
        "INPUT_SIZE": INPUT_SIZE,
        "HIDDEN_SIZE": HIDDEN_SIZE,
        "OUTPUT_SIZE": OUTPUT_SIZE,
        "MIDDLE_LAYERS": MIDDLE_LAYERS,
        "device": device.type
    }
    save_model(model, model_subdir, subdir_num, params, "windbrain.pth")
    #save_parameters(model_subdir, subdir_num, params)
    #def save_model(model, model_subdir, subdir_num, params, model_filename="seedbrain.pth"):

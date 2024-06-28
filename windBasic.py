''' 
Starting with an "MVP" deep q network
Now making it a wind brain one step at a time
'''


import torch
import torch.nn as nn 
import torch.nn.functional as F
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

from game import BOARD_WIDTH, BOARD_HEIGHT, initialize_board, dir_pairs #, place_dandelion, spread_seeds, check_dandelion_win, convert_user_input, direction_names, validate_row_input, validate_col_input, validate_direction_input, play_game,  display_board_with_labels


## BOARD_HEIGHT = BOARD_WIDTH = 5  # will always be 5


LEARNING_RATE = 0.002
# learning rate is steped down in the code scheduler = ReduceLROnPlateau...

#EXPLORATION_PROB = 0.01
EXPLORATION_PROB_STEPS = {20:0.2,   # first x percent of epochs -> y
                          45:0.001,  # after x percent, etc
                          50:0}     

NUM_DIR = 8 # will always be 8, but here for clarity


INPUT_SIZE = NUM_DIR + 2 *(BOARD_HEIGHT * BOARD_WIDTH)
HIDDEN_SIZE = INPUT_SIZE *2
OUTPUT_SIZE = NUM_DIR

EPOCHS = 10000

# Gamma aka discount factor for future rewards or "Decay"
GAMMA = 0.97  

reward_vals = {
    "win": 100, 
    "illegal": -100, 
    "lose": -100,
    "meh": 0
}


wcnt=0

#good_pass_cnt = 0

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        logits = F.relu(self.fc1(x))
        logits = self.fc2(logits)
        return logits

model = DQN()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=EPOCHS//100, cooldown=EPOCHS//10)


def game_step(state, action):

    dir_list, board_grid = board_state_from_tensor(state)
    # print(f" dir_list: {dir_list=}")
    # print(f" board_grid: {board_grid=}")

    if dir_list[action] == 1:
        return state, reward_vals['illegal'], 1  # Illegal move
    new_dir_list = dir_list[:]
    new_dir_list[action] = 1


    # spread seeds(action)
    ##def spread_seeds(board, direction_tuple):

    new_state = board_state_to_tensor(new_dir_list, board_grid )
    # check dandelion win
    dandelion_win = False # for now...  check_dandelion_win(board)
    if dandelion_win:
        return new_state, reward_vals['lose'], 1  # Game lost

    # still here?
    if sum(new_dir_list) >= 7:  # wind has gone 7 times. Winner!
        return new_state, reward_vals['win'], 1  # Game won
    else:
        #print("keep going")
        #print(f"{new_state=}")
        return new_state, reward_vals['meh'], 0  # Continue game



def board_state_to_tensor(direction_list, board):
    # 1st 8 cells are available directions
    # next 25 cells are dandelions
    # next 25 cells are seeds
    direction_tensor = torch.tensor([1 if direction else 0 for direction in direction_list]).float()
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


#        dir_tuple = dir_pairs[direction_names.index(chosen_direction)]
#        board = spread_seeds(board, dir_tuple)


############### 
# begin
############### 

rec_loss_mod, rec_loss_idx = 16, 0
rec_loss = [0] * rec_loss_mod

for epoch in range(EPOCHS):

    used_dirs = [0] * NUM_DIR  # Initialize game state
    board_grid = initialize_board()
    # todo, have seeds make first move
    state_tensor = board_state_to_tensor(used_dirs, board_grid)





    run_percent = (epoch + 1) / EPOCHS * 100
    for percent, prob in EXPLORATION_PROB_STEPS.items():
        if run_percent < percent:
            EXPLORATION_PROB = prob
            break

    done = False

    while not done:
        # Generate predictions
        q_values_pred = model(state_tensor)   # calls forward
        q_values_pred = q_values_pred.unsqueeze(0)  # get correct tensor shape

        
        # init target_q_values with zeroes
        target_q_values = torch.zeros(1, NUM_DIR)

        # explore all actions (exhaustive checking for given state)
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


        ## this should happen per step (not per epoch)
        optimizer.zero_grad()
        loss = F.mse_loss(q_values_pred, target_q_values) # oringinal code (whole tensor)
        loss.backward()
        optimizer.step()



        # prep for next iteration
        if random.random() < EXPLORATION_PROB:      # explore? 
            #expl = 'explore'
            action = random.randint(0, NUM_DIR-1)
        else:                                       # else exploit
            #expl = 'exploit'
            action = torch.argmax(q_values_pred).item()

        #print("next action ", action)
        next_state, reward, done = game_step(state_tensor, action)
        state_tensor = next_state

        if reward == reward_vals['win']:
            wcnt += 1
            print("W", end="")


        # if epoch % 244 == 0:
        #     print(f"\n\n\n\n         {action=}")
        #     print(f"           {expl=}")
        #     print(f"     {orig_state=}")
        #     print(f"(after a) {state=}")
        #     print(f"{  q_values_pred=}\n")
        #     print(f"{target_q_values=} \n")
        #     if epoch > 14400:
        #          quit()


    #print(f"^^^^^^^^^^^         {epoch=}        RUN DONE   ^^^^^^^^^^^^^^^^^^^^^\n\n")


    rec_loss[rec_loss_idx] = loss.item()
    rec_loss_idx += 1
    if rec_loss_idx % rec_loss_mod == 0:
        rec_loss_idx = 0



    # update the Learning Rate if the loss has platued
    if epoch > 2* rec_loss_mod and run_percent > 50 and loss.item() < 100:
       avg_loss = sum(rec_loss) / rec_loss_mod
       scheduler.step(avg_loss)
    ##scheduler.step(loss)


    if (epoch ) % 100 == 0:
        learn_rate = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}, run_perc {round(run_percent, 1)} {wcnt=} {EXPLORATION_PROB=} {learn_rate=} Most Recent Loss: {loss.item()}\n{q_values_pred=}")
        #print(f"  {q_values_pred=}\n{target_q_values=}\n")
        print(f"recent loss avg: {sum(rec_loss)/rec_loss_mod}")


        '''
        if wcnt > 99:
            good_pass_cnt += 1
        if good_pass_cnt > 5:
            print("got good by epoch", epoch +1)
            #quit()
        '''
        wcnt = 0
''' 
Starting with an "MVP" deep q network
Now making it a wind brain one step at a time
'''


# very much a WIP, checking in, gotta run, would like to clean up more before check in but gotta run


import torch
import torch.nn as nn 
import torch.nn.functional as F
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

from game import BOARD_WIDTH, BOARD_HEIGHT, initialize_board, display_board_with_labels, place_dandelion, spread_seeds, check_dandelion_win, convert_user_input, dir_pairs, direction_names, validate_row_input, validate_col_input, validate_direction_input, play_game



BOARD_HEIGHT = BOARD_WIDTH = 5  # will always be 5


LEARNING_RATE = 0.002

#EXPLORATION_PROB = 0.01
EXPLORATION_PROB_STEPS = {15:0.2,   # first x percent of epochs -> y
                          45:0.001,  # after x percent, etc
                          50:0}     

NUM_DIR = 8 # will always be 8, but here for clarity
INPUT_SIZE = NUM_DIR
HIDDEN_SIZE = INPUT_SIZE *2
OUTPUT_SIZE = NUM_DIR

EPOCHS = 20000

# Gamma aka discount factor for future rewards or "Decay"
GAMMA = 0.97  

reward_vals = {
    "win": 100, 
    "illegal": -100, 
    "lose": -100,
    "meh": 1
}

wcnt=0

good_pass_cnt = 0

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
    #print(f"\nstate: {state}, action: {action}")
    if state[action] == 1:
        #moves_made = sum(state)
        #print(f"ILLEGAL MOVE      {moves_made=} {'-' * moves_made}")
        return state, reward_vals['illegal'], 1  # Illegal move
    else:
        new_state = state[:]
        new_state[action] = 1
        if sum(new_state) >= 7:
            return new_state, reward_vals['win'], 1  # Game won
        else:
            #print("keep going")
            return new_state, reward_vals['meh'], 0  # Continue game



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

############### 
# begin
############### 

rec_loss_mod = 16
rec_loss_idx = 0
rec_loss = [0] * rec_loss_mod

for epoch in range(EPOCHS):
    state = [0] * NUM_DIR  # Initialize game state

    ''' 
    If initial game state is far from win, it usually gets stuck in a bad state
    I think this is a local optimum
    Update: fixed with explore v exploit
    '''


    run_percent = (epoch + 1) / EPOCHS * 100
    for percent, prob in EXPLORATION_PROB_STEPS.items():
        if run_percent < percent:
            EXPLORATION_PROB = prob
            break

    done = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
        
        #print(f"{state=}  {q_values=}")

        # begin exhaustive here
        orig_state = state
        # init target_q_values to zeroes
        target_q_values = torch.zeros(1, NUM_DIR)

        for action in range(NUM_DIR):
            next_state, reward, done = game_step(orig_state, action)


            state = next_state
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            with torch.no_grad():
                next_q_values = model(next_state_tensor)

            max_next_q_value = torch.max(next_q_values).item()
            if done:
                bellman_right = reward
            else:
                bellman_right = reward + GAMMA * max_next_q_value  # This is the target value



            # is this used? bellman_left = q_values[0, action]  # This is the current estimate from the network
            #print(f"{bellman_left=} {bellman_right=} \n     {q_values=}\n{next_q_values=}\n{reward=}\n\n")

            # Learning step
            optimizer.zero_grad()  # should this be here???
            q_values_pred = model(state_tensor)


            #target_q_values = q_values_pred.clone()
            target_q_values[0, action] = bellman_right  # Update using Bellman equation




        # end exhaustive
        '''
        # non-exhaustive
        if random.random() < EXPLORATION_PROB:      # explore? 
            action = random.randint(0, NUM_DIR-1)
        else:                                       # else exploit
            action = torch.argmax(q_values).item()

        next_state, reward, done = game_step(state, action)
        #run_reward += reward

        # Prepare for next iteration
        state = next_state
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        with torch.no_grad():
            next_q_values = model(next_state_tensor)

        
        # bellman_right should just be the reward if it's in a terminal state
        if done:
            bellman_right = reward
        else:
            max_next_q_value = torch.max(next_q_values).item()
            bellman_right = reward + done * GAMMA * max_next_q_value  # This is the target value
        


        bellman_left = q_values[0, action]  # This is the current estimate from the network
        #print(f"{bellman_left=} {bellman_right=} \n     {q_values=}\n{next_q_values=}\n{reward=}\n\n")

        # Learning step
        optimizer.zero_grad()
        q_values_pred = model(state_tensor)


        target_q_values = q_values_pred.clone()
        target_q_values[0, action] = bellman_right  # Update using Bellman equation

        # end non-exhaustive

        if epoch % 244 == 0:
            print(f"\n\n\n\n         {action=}")
            print(f"(after a) {state=}")
            print(f"   {bellman_left=}")
            print(f"{  q_values_pred=}\n")
            print(f"  {bellman_right=}")
            print(f"{target_q_values=} \n")

            if epoch > 4400:
                quit()

        '''

        # exhaustive prep for next iteration
        if random.random() < EXPLORATION_PROB:      # explore? 
            #print("explore")
            expl = 'explore'
            action = random.randint(0, NUM_DIR-1)
        else:                                       # else exploit
            expl = 'exploit'
            #print("exploit")
            action = torch.argmax(q_values).item()

        #print("next action ", action)
        next_state, reward, done = game_step(orig_state, action)
        state = next_state

        if reward == reward_vals['win']:
            #global wcnt
            wcnt += 1
            print("W", end="")


        '''
        if epoch % 244 == 0:
            print(f"\n\n\n\n         {action=}")
            print(f"           {expl=}")
            print(f"     {orig_state=}")
            print(f"(after a) {state=}")
            print(f"   {bellman_left=}")
            print(f"{  q_values_pred=}\n")
            print(f"  {bellman_right=}")
            print(f"{target_q_values=} \n")

            if epoch > 4400:
                quit()

        if epoch % 244 == 0:
            print ("DO LOSS BACKWARD AND OPTIMZER STEP")
        '''

        optimizer.zero_grad()  # should this be here???

        ## this should happen per step (not per epoch)
        loss = F.mse_loss(q_values_pred, target_q_values) # oringinal code (whole tensor)
        loss.backward()
        optimizer.step()

    #print(f"^^^^^^^^^^^         {epoch=}        RUN DONE   ^^^^^^^^^^^^^^^^^^^^^\n\n")


    rec_loss[rec_loss_idx] = loss.item()
    rec_loss_idx += 1
    if rec_loss_idx % rec_loss_mod == 0:
        rec_loss_idx = 0



    # update the Learning Rate if the loss has platued
    if epoch > 2* rec_loss_mod and run_percent > 50 and loss.item() < 100:
        avg_loss = sum(rec_loss) / rec_loss_mod
        scheduler.step(avg_loss)
    ###scheduler.step(loss)


    if (epoch ) % 100 == 0:

        learn_rate = optimizer.param_groups[0]['lr']


        print(f"\nEpoch {epoch}, run_perc {round(run_percent, 1)} {wcnt=} {EXPLORATION_PROB=} {learn_rate=} Most Recent Loss: {loss.item()}\n{q_values=}\n")
        #print(f"  {q_values_pred=}\n{target_q_values=}\n{state=}\n")
        print(f"recent loss avg: {sum(rec_loss)/rec_loss_mod}")


        '''
        if wcnt > 99:
            good_pass_cnt += 1
        if good_pass_cnt > 5:
            print("got good by epoch", epoch +1)
            #quit()
        '''
        wcnt = 0
        
    #if epoch > 99:
    #    quit()


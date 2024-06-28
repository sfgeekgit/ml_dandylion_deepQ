''' 
Starting with an "MVP" deep q network
Now making it a wind brain one step at a time
'''


import torch
import torch.nn as nn 
import torch.nn.functional as F
import random

LEARNING_RATE = 0.005

#EXPLORATION_PROB = 0.01
EXPLORATION_PROB_STEPS = {15:0.2,    # first 15 percent of epochs -> .1
                          25:0.001,  # after 25 percent of epochs -> .001
                          50:0}     # after 50 percent of epochs -> 0

NUM_DIR = 8 # will always be 8, but here for clarity
INPUT_SIZE = NUM_DIR
HIDDEN_SIZE = INPUT_SIZE *2
OUTPUT_SIZE = NUM_DIR

EPOCHS = 50000

# Gamma aka discount factor for future rewards or "Decay"
GAMMA = 0.97  

reward_vals = {
    "win": 100, 
    "illegal": -100, 
    "lose": -100,
    "meh": 0
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
    I think this currently does 100% exploit, no exploration
    TODO: set some explore

    '''

    # flip a random number of bit in the initial state
    #for _ in range(random.randint(0, NUM_DIR-1)):
    #    state[random.randint(0, NUM_DIR-1)] = 1

    #print ("\n\n\n\n Init count: ", sum(state))

    #run_reward = 0
    done = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
        
        #print(f"{state=}  {q_values=}")


        # try exhaustive here?
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



            bellman_left = q_values[0, action]  # This is the current estimate from the network
            #print(f"{bellman_left=} {bellman_right=} \n     {q_values=}\n{next_q_values=}\n{reward=}\n\n")

            # Learning step
            optimizer.zero_grad()
            q_values_pred = model(state_tensor)


            #target_q_values = q_values_pred.clone()
            target_q_values[0, action] = bellman_right  # Update using Bellman equation

            
            if epoch % 244 == 0:
                print(f"     {orig_state=}")
                print(f"         {action=}")
                print(f"   {bellman_left=}")
                print(f"{  q_values_pred=}")
                print(f"         {reward=}")
                print(f"           {done=}")
                print(f"  {bellman_right=}")
                print(f"{target_q_values=} \n\n\n")
            


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


        print(f"{action=}")
        print(f"   {bellman_left=}")
        print(f"{  q_values_pred=}\n")
        print(f"  {bellman_right=}")
        print(f"{target_q_values=} \n\n\n")
        '''



        loss = F.mse_loss(q_values_pred, target_q_values) # oringinal code (whole tensor)
        rec_loss[rec_loss_idx] = loss.item()
        rec_loss_idx += 1
        if rec_loss_idx % rec_loss_mod == 0:
            rec_loss_idx = 0


        #loss = F.mse_loss(bellman_left, bellman_right) # one number (bellman_right)
        #optimizer.zero_grad()  # need this if loss is just bellman numbers, not whole tensor

        #print (f"--\n\nNOW LOSS\n{loss.item()=}\n\n")

        loss.backward()
        optimizer.step()


        run_percent = (epoch + 1) / EPOCHS * 100
        for percent, prob in EXPLORATION_PROB_STEPS.items():
            if run_percent < percent:
                EXPLORATION_PROB = prob
                break
        # exhaustive prep for next iteration
        if random.random() < EXPLORATION_PROB:      # explore? 
            #print("explore")
            action = random.randint(0, NUM_DIR-1)
        else:                                       # else exploit
            #print("exploit")
            action = torch.argmax(q_values).item()

        #print("next action ", action)
        next_state, reward, done = game_step(orig_state, action)
        state = next_state

        if reward == reward_vals['win']:
            #global wcnt
            wcnt += 1
            print("W", end="")


        #if done:
            #print ("\nterminal choice!\n")
        #print(f"{next_state=}\n\n\n")

    #print(f"^^^^^^^^^^^         {epoch=}        RUN DONE   ^^^^^^^^^^^^^^^^^^^^^\n\n")


    if (epoch + 1) % 100 == 0:
        print(f"\nEpoch {epoch+1}, {wcnt=} {EXPLORATION_PROB=} Most Recent Loss: {loss.item()}\n{q_values=}\n")
        #print(f"  {q_values_pred=}\n{target_q_values=}\n{state=}\n")
        print(f"recent loss avg: {sum(rec_loss)/rec_loss_mod}")


        
        if wcnt > 99:
            good_pass_cnt += 1
        if good_pass_cnt > 5:
            print("got good by epoch", epoch +1)
            #quit()

        #quit()
        wcnt = 0

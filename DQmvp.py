''' 
Starting with an "MVP" deep q network
Adding and experimenting and tweaking it
By the time you read this, it may or may not still be an MVP
Maybe should rename to like "playground" or something
'''


import torch
import torch.nn as nn 
import torch.nn.functional as F
import random

LEARNING_RATE = 0.005

EXPLORATION_PROB = 0.01

NUM_DIR = 8 # will always be 8, but here for clarity
INPUT_SIZE = NUM_DIR
HIDDEN_SIZE = INPUT_SIZE *2
OUTPUT_SIZE = NUM_DIR

EPOCHS = 25000

# Gamma aka discount factor for future rewards or "Decay"
GAMMA = 0.95  

reward_vals = {
    "win": 100, 
    "illegal": -100, 
    "lose": -100,
    "meh": 10
}

wcnt=0

streak = 0

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
        return state, reward_vals['illegal'], True  # Illegal move
    else:
        new_state = state[:]
        new_state[action] = 1
        if sum(new_state) == 7:
            global wcnt
            wcnt += 1
            print("W", end="")
            return new_state, reward_vals['win'], True  # Game won
        else:
            #print("keep going")
            return new_state, reward_vals['meh'], False  # Continue game


############### 
# begin
############### 

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

    run_reward = 0
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
        
        #print(f"{state=}  {q_values=}")


        if random.random() < EXPLORATION_PROB:      # explore? 
            action = random.randint(0, NUM_DIR-1)
        else:                                       # else exploit
            action = torch.argmax(q_values).item()

        next_state, reward, done = game_step(state, action)
        run_reward += reward

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
            bellman_right = reward + GAMMA * max_next_q_value  # This is the target value



        bellman_left = q_values[0, action]  # This is the current estimate from the network
        #print(f"{bellman_left=} {bellman_right=} \n     {q_values=}\n{next_q_values=}\n{reward=}\n\n")

        # Learning step
        optimizer.zero_grad()
        q_values_pred = model(state_tensor)


        target_q_values = q_values_pred.clone()
        target_q_values[0, action] = bellman_right  # Update using Bellman equation

        '''
        print(f"{action=}")
        print(f"   {bellman_left=}")
        print(f"{  q_values_pred=}\n")
        print(f"  {bellman_right=}")
        print(f"{target_q_values=} \n\n\n")
        '''

        loss = F.mse_loss(q_values_pred, target_q_values) # oringinal code (whole tensor)
        #loss = F.mse_loss(bellman_left, bellman_right) # one number (bellman_right)
        #optimizer.zero_grad()  # need this if loss is just bellman numbers, not whole tensor

        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"\nEpoch {epoch+1}, {wcnt=} Most Recent Loss: {loss.item()}, This Run Reward: {run_reward}\n{q_values=}\n")
        if wcnt > 95:
            streak += 1
        if streak > 4:
            print("got good by epoch", epoch +1)
            quit()

        wcnt = 0



import os
import torch
#from game import initialize_board, display_board_with_labels, place_dandelion, spread_seeds, check_dandelion_win, convert_user_input, dir_pairs, direction_names, validate_row_input, validate_col_input, validate_direction_input, BOARD_WIDTH, BOARD_HEIGHT, get_human_wind_move
from game import *
#from seedBrain import DQN
from brainlib import * # board_state_to_tensor #, board_state_from_tensor


'''
Just the beginning of model vs model play
To do: loop through models, collect results, plot, etc.
Make it so a model can train against another model move by move (i.e. a self play loop) currently it just playes whole games
add a "temperature" to add some randomness to the moves for both models
I have some ideas how I want to apply temperature, but should maybe look into literature on the topic
'''

seedbrain_dir      = "./models/seeds/" 
windbrain_dir =      "./models/wind/"

stochastic_logging = False
if stochastic_logging:
    seed_moves_cnt ={'best':0, 'rest':0}
    seed_moves_tally = []
    wind_moves_cnt ={'best':0, 'rest':0}
    wind_moves_tally = []


def model_v_model(seedbrain, windbrain, seed_temp=0, wind_temp=0):
    # temperature of 0 means it's deterministic, 1 is completely random. Generally use 0 or near 0
    board = initialize_board()
    used_directions = [0] * len(direction_names)

    winner = None

    seed_temp, wind_temp = 4.0, 4.0

    # Main game loop
    for turn in range(7):
        # Seedbrain makes a move
        row, col = seedbrain_move_stochastic(used_directions, board, seedbrain, seed_temp)# , temperature=0.0)

        # Check if the board already has a dandelion at that location
        if board[row][col] == 1:
            print(f"Dandelion already placed at {row}, {col}")
            print("\n!!!!!!!!!!! Dandelions Forfeit!! Wind wins!!")
            winner = "w"
            break
        place_dandelion(board, row, col)

        # Check for immediate win condition
        if check_dandelion_win(board):
            print('Dandelions win! Plant in last empty spot!')
            winner = "s"
            break


        # Windbrain makes a move
        dir_names = direction_names.copy() # is this necessary?
        for i in range(len(dir_names)):
            if used_directions[i] == 1:
                dir_names[i] = i
        ### print(f"# Avail Directions: {dir_names}")
        used_dir_before = used_directions.copy()
        dir_tuple = windbrain_move_stochastic(used_directions, board, windbrain, wind_temp)
        if used_dir_before == used_directions:
            print("Wind reused a direction! Wind forfeits!")
            winner = "s"
            break
            
        board = spread_seeds(board, dir_tuple)
        display_board_with_labels(board)


    print("Game over.")
    if stochastic_logging:
        print(f"Seed moves: {seed_moves_cnt} {''.join(seed_moves_tally)}")
        print(f"Wind moves: {wind_moves_cnt} {''.join(wind_moves_tally)}")

    if winner:
        print("Final board:")
        display_board_with_labels(board)
        print(f"{winner=}")
    else:
        if check_dandelion_win(board):
            winner = "s"
            print("!!!!!!!!!!! Dandelions win!!!")
        else:
            winner = "w"
            print('!!!!!!!!!!! Wind wins!!!!!!!!')
    return winner

if __name__ == "__main__":
    # Load models
    seedbrain_dir = "models/seeds/007/"
    seedbrain_filename = "seedbrain.pth"
    windbrain_dir = "models/wind/014/"
    windbrain_filename = "windbrain.pth"

    seedbrain = load_model(seedbrain_dir, seedbrain_filename)
    windbrain = load_model(windbrain_dir, windbrain_filename)

    # Run model vs model game
    model_v_model(seedbrain, windbrain)

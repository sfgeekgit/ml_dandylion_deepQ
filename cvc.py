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


def seedbrain_move(used_dirs, board, model, temperature=11.0):
    board_tensor = board_state_to_tensor(used_dirs, board, device=torch.device("cpu"))
    with torch.no_grad():
        q_values = model(board_tensor)

    ideal_move_idx = torch.argmax(q_values).item()
    move_idx = select_action_with_temperature(q_values, temperature)

    row_label, col_label = seed_idx_to_label(move_idx)
    if move_idx != ideal_move_idx:
        ideal_row_label, ideal_col_label = seed_idx_to_label(ideal_move_idx)
        print(f"Seed Best Move: {ideal_row_label}, {ideal_col_label} selected: {row_label}, {col_label}")
        print(f"Seed MoveQ values: {q_values}\n\n\n\n")
        #quit()


    print(f"Seed Move: {row_label}, {col_label}  {temperature=}")

    return move_idx // BOARD_WIDTH, move_idx % BOARD_WIDTH

def windbrain_move(used_dirs, board, model, temperature=0):
    board_tensor = board_state_to_tensor(used_dirs, board, device=torch.device("cpu"))
    with torch.no_grad():
        q_values = model(board_tensor)
    print(f"Wind MoveQ values: {q_values}")
    chosen_direction = torch.argmax(q_values).item()
    print(f"Wind Move: {chosen_direction}")

    #dir_tuple = dir_pairs[direction_names.index(chosen_direction)]
    dir_tuple = dir_pairs[chosen_direction]
    used_dirs[chosen_direction] = 1
    return dir_tuple

def model_v_model(seedbrain, windbrain, seed_temp=0, wind_temp=0):
    # temperature of 0 means it's deterministic, 1 is completely random. Generally use 0 or near 0
    board = initialize_board()
    used_directions = [0] * len(direction_names)

    winner = None

    # Main game loop
    for turn in range(7):
        # Seedbrain makes a move
        row, col = seedbrain_move(used_directions, board, seedbrain)

        # Check if the board already has a dandelion at that location
        if board[row][col] == 1:
            print(f"Dandelion already placed at {row}, {col}")
            print("\n!!!!!!!!!!! Dandelions Forfeit!! Wind wins!! But.. let's keep playing anyway.\n")
            winner = "w"
            print("no lets stop")
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
        print(f"# Avail Directions: {dir_names}")
        used_dir_before = used_directions.copy()
        dir_tuple = windbrain_move(used_directions, board, windbrain)
        if used_dir_before == used_directions:
            print("Wind reused a direction! Wind forfeits!")
            winner = "s"
            break
            
        board = spread_seeds(board, dir_tuple)
        display_board_with_labels(board)


    print("Game over.")
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

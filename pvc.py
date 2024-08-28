import os
import torch
#from game import initialize_board, display_board_with_labels, place_dandelion, spread_seeds, check_dandelion_win, convert_user_input, dir_pairs, direction_names, validate_row_input, validate_col_input, validate_direction_input, BOARD_WIDTH, BOARD_HEIGHT, get_human_wind_move
from game import *
#from seedBrain import DQN
from brainlib import * # board_state_to_tensor #, board_state_from_tensor

wind_human = True
wind_human = False

seed_human = True
seed_human = False


def seedbrain_move(used_dirs, board, model):
    board_tensor = board_state_to_tensor(used_dirs, board, device=torch.device("cpu"))
    with torch.no_grad():
        q_values = model(board_tensor)
    best_move_idx = torch.argmax(q_values).item()
    return best_move_idx // BOARD_WIDTH, best_move_idx % BOARD_WIDTH

def windbrain_move(used_dirs, board, model):
    board_tensor = board_state_to_tensor(used_dirs, board, device=torch.device("cpu"))
    with torch.no_grad():
        q_values = model(board_tensor)
    print(f"Wind MoveQ values: {q_values}")
    chosen_direction = torch.argmax(q_values).item()
    print(f"Wind Best Move: {chosen_direction}")

    #dir_tuple = dir_pairs[direction_names.index(chosen_direction)]
    dir_tuple = dir_pairs[chosen_direction]
    used_dirs[chosen_direction] = 1
    return dir_tuple


def play_game_against_model():

    # Hardcode the model directory for now
    seedbrain_dir     = "models/seeds/006/" 
    seedbrain_filename =  "seedbrain.pth"

    windbrain_dir = "models/wind/014/"
    windbrain_filename = "windbrain.pth"


    if not seed_human:
        seedbrain = load_model(seedbrain_dir, seedbrain_filename)
        print(f"Loaded model from {seedbrain_dir}")


    if not wind_human:
        windbrain = load_model(windbrain_dir, windbrain_filename)
        print(f"Loaded model from {windbrain_dir}")

    board = initialize_board()
    #available_direction_names = direction_names.copy()
    used_directions = [0] * len(direction_names)

    #avail_directions = [1] * 8
    #print(f"Available directions: {avail_directions}")

    winner = None
    # Main game loop
    for turn in range(7):        
        # Seedbrain makes a move
        if seed_human:
            row, col = get_human_seed_move(board, used_directions)
        else:
            row, col = seedbrain_move(used_directions, board, seedbrain)

        # check if the board already has a dandelion at that location
        if board[row][col] == 1:
            print(f"Dandelion already placed at {row}, {col}")
            print("\n!!!!!!!!!!! Dandelions Forfeit!! Wind wins!! But.. let's keep playing anyway.\n")
            winner = "w"
        place_dandelion(board, row, col)

        # Check for immediate win condition
        if check_dandelion_win(board):
            print('Dandelions win!')
            winner = "s"
            break
        
        if wind_human:
            print("Wind's turn. Try to NOT spread the seeds.")
            # get_human_wind_move function updates used_directions I'm not used to python working that way, but... here we are.
            dir_tuple = get_human_wind_move(board, used_directions)  
        else:
            #print(f"Before move: {used_directions=}")
            #print("# Directions (N, S, E, W, NE, NW, SE, SW)")
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
    play_game_against_model()
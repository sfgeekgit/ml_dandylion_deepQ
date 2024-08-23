import os
import torch
#from game import initialize_board, display_board_with_labels, place_dandelion, spread_seeds, check_dandelion_win, convert_user_input, dir_pairs, direction_names, validate_row_input, validate_col_input, validate_direction_input, BOARD_WIDTH, BOARD_HEIGHT, get_human_wind_move
from game import *
#from seedBrain import DQN
from brainlib import board_state_to_tensor #, board_state_from_tensor

wind_human = True
wind_human = False

seed_human = True
seed_human = False

def load_model(model_path, model_type="seed"):
    if model_type == "seed":
        import seedBrain
        model = seedBrain.DQN()
        # problem! This requires the current seedbrain.py code to have the same shape as the saved model.

    elif model_type == "wind":
        import windBrain
        model = windBrain.DQN()
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
    model.eval()
    return model

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
    seedbrain_path = "models/010/seedbrain.pth" # high trained
    #seedbrain_path = "models/027/seedbrain.pth" # low trained

    windbrain_path = "models/wind/007/windbrain.pth" # high trained
    #windbrain_path = "models/wind/001/windbrain.pth" # low trained


    if not seed_human:
        seedbrain = load_model(seedbrain_path, model_type="seed")
        print(f"Loaded model from {seedbrain_path}")

    if not wind_human:
        windbrain = load_model(windbrain_path, model_type="wind")
        print(f"Loaded model from {windbrain_path}")

    board = initialize_board()
    #available_direction_names = direction_names.copy()
    used_directions = [0] * len(direction_names)

    #avail_directions = [1] * 8
    #print(f"Available directions: {avail_directions}")

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

        place_dandelion(board, row, col)

        # Check for immediate win condition
        if check_dandelion_win(board):
            print('Dandelions win!')
            break
        
        if wind_human:
            print("Wind's turn. Try to NOT spread the seeds.")
            # get_human_wind_move function updates used_directions I'm not used to python working that way, but... here we are.
            dir_tuple = get_human_wind_move(board, used_directions)  
        else:
            print(f"Before move: {used_directions=}")
            dir_tuple = windbrain_move(used_directions, board, windbrain)
            print(f"After move: {used_directions=}")
        
        board = spread_seeds(board, dir_tuple)

    display_board_with_labels(board)
    print("Game over.")
    # Check for win condition if the game wasn't already won
    if check_dandelion_win(board):
        print("!!!!!!!!!!! Dandelions win!!!")
    else:
        print('!!!!!!!!!!! Wind wins!!!!!!!!')

if __name__ == "__main__":
    play_game_against_model()
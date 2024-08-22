import os
import torch
#from game import initialize_board, display_board_with_labels, place_dandelion, spread_seeds, check_dandelion_win, convert_user_input, dir_pairs, direction_names, validate_row_input, validate_col_input, validate_direction_input, BOARD_WIDTH, BOARD_HEIGHT, get_human_wind_move
from game import *
from seedBrain import DQN
from brainlib import board_state_to_tensor #, board_state_from_tensor

# TODO
# check the available directions and pass them to the model
# the tensor needs to be feed the USED direction list

def load_model(model_dir):
    model = DQN()
    model_path = os.path.join(model_dir, "seedbrain.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def seedbrain_move(used_dirs, board, model):
    # Convert the board to a tensor and get the model's move
    # board_state_to_tensor needs list of USED directions.

    board_tensor = board_state_to_tensor(used_dirs, board, device=torch.device("cpu"))
    with torch.no_grad():
        q_values = model(board_tensor)
    best_move_idx = torch.argmax(q_values).item()
    return best_move_idx // BOARD_WIDTH, best_move_idx % BOARD_WIDTH

def play_game_against_seedbrain():
    print("\n\nStarting game with seedbrain. You are the wind. Try to NOT spread the seeds.\n\nGood luck!\n")

    # Hardcode the model directory for now
    model_dir = "models/010/" # high trained
    #model_dir = "models/027/" # low trained
    seedbrain = load_model(model_dir)
    print(f"Loaded model from {model_dir}")


    board = initialize_board()
    available_direction_names = direction_names.copy()
    used_directions = [0] * len(direction_names)

    #avail_directions = [1] * 8
    #print(f"Available directions: {avail_directions}")

    # Main game loop
    for turn in range(7):        
        # Seedbrain makes a move
        print(f"Available directions: {available_direction_names}")
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
        
        print("Wind's turn. Try to NOT spread the seeds.")
        #dir_tuple = get_human_wind_move(board, available_direction_names)
        dir_tuple = get_human_wind_move(board, used_directions)
        board = spread_seeds(board, dir_tuple)
    display_board_with_labels(board)
    print("Game over.")
    # Check for win condition if the game wasn't already won
    if check_dandelion_win(board):
        print("!!!!!!!!!!! Dandelions win!!!")
    else:
        print('!!!!!!!!!!! Wind wins!!!!!!!!')

if __name__ == "__main__":
    play_game_against_seedbrain()
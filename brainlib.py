import torch
import os

from game import BOARD_WIDTH, BOARD_HEIGHT, NUM_DIR

def board_state_to_tensor(used_direction_list, board, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # 1st 8 cells are available directions
    # next 25 cells are dandelions
    # next 25 cells are seeds
    direction_tensor = torch.tensor([1 if direction else 0 for direction in used_direction_list], dtype=torch.float32).to(device)
    if BOARD_HEIGHT == 0:  # dev
        return direction_tensor 
    dandelion_tensor  = torch.tensor([[1 if cell == 1 else 0 for cell in row] for row in board], dtype=torch.float32).to(device)
    seed_tensor       = torch.tensor([[1 if cell == 2 else 0 for cell in row] for row in board], dtype=torch.float32).to(device)
    return torch.cat((direction_tensor, dandelion_tensor.view(-1), seed_tensor.view(-1)))


def board_state_from_tensor(tensor, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # 1st 8 cells are available directions ## update -- used directions
    # next 25 cells are dandelions
    # next 25 cells are seeds
    used_direction_list = [1 if direction else 0 for direction in tensor[:NUM_DIR]]
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
    return [used_direction_list, board]


def get_next_model_subdir(base_dir="models"):
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)

    existing_subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    numeric_subdirs = sorted([int(d) for d in existing_subdirs if d.isdigit()])
    if numeric_subdirs:
        next_subdir_num = numeric_subdirs[-1] + 1
    else:
        next_subdir_num = 1

    # Format the next subdirectory name
    next_subdir_name = f"{next_subdir_num:03d}"
    
    return os.path.join(base_dir, next_subdir_name), next_subdir_num

def save_parameters(subdir, subdir_num, params):
    params_filename = f"params{subdir_num:03d}.py"
    params_filepath = os.path.join(subdir, params_filename)
    
    with open(params_filepath, 'w') as f:
        for key, value in params.items():
            f.write(f"{key} = {repr(value)}\n")
    print(f"Parameters saved to {params_filepath}")

import torch
import os
from torch import nn

from game import BOARD_WIDTH, BOARD_HEIGHT, NUM_DIR, dir_pairs, seed_idx_to_label

stochastic_logging = False

class DQN(nn.Module):  
    def __init__(self, input_size, hidden_sizes, output_size, device='cpu'):
        super(DQN, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.model = nn.Sequential(*layers)
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.model(x)
    

def select_action_with_temperature(q_values, temperature=0.0):
    if temperature == 0:
        # Deterministic case: return the index of the maximum q_value
        return [torch.argmax(q_values).item(), True]
    else:
        scaled_q_values = q_values / temperature
        probabilities = torch.softmax(scaled_q_values, dim=0)
        action_index = torch.multinomial(probabilities, num_samples=1)

        
        if action_index.item() == torch.argmax(q_values).item():
            pbest = True
            return [action_index.item(), True]
            #print ("\n\n\n\n\n---------\n\n Picked the best move")
        else:
            pbest = False
            #print("\n\n\n\n\n---------\n\n Picked a different move")

        #stochastic_logging = False
        global stochastic_logging
        if stochastic_logging:
            formatted_probabilities = [f"{100*p:.1f}" for p in probabilities.tolist()]
            print(f"q_values: {q_values} {len(q_values)=}")
            print(f"Probabilities: {formatted_probabilities=}")
            print(f"Highest probability: {100*max(probabilities.tolist()):.1f}%")
            print(f"odds this choice was selected: {formatted_probabilities[action_index.item()]}\n\n\n\n")

    
        return [action_index.item(), pbest]
    

def seedbrain_move_stochastic(used_dirs, board, model, temperature=4.0):
    board_tensor = board_state_to_tensor(used_dirs, board, device=torch.device("cpu"))
    with torch.no_grad():
        q_values = model(board_tensor)

    ideal_move_idx = torch.argmax(q_values).item()
    [move_idx, picked_best] = select_action_with_temperature(q_values, temperature)
    row_label, col_label = seed_idx_to_label(move_idx)

    stochastic_logging = False
    if stochastic_logging:
        if picked_best:
            seed_moves_cnt['best'] += 1
            seed_moves_tally.append('_')
        else:
            seed_moves_cnt['rest'] += 1
            seed_moves_tally.append('X')

        if move_idx != ideal_move_idx:
            ideal_row_label, ideal_col_label = seed_idx_to_label(ideal_move_idx)
            print(f"Seed Best Move: {ideal_row_label}, {ideal_col_label} selected: {row_label}, {col_label}")

    print(f"--------------  Seed Move: {row_label}, {col_label}  {temperature=}")

    return move_idx // BOARD_WIDTH, move_idx % BOARD_WIDTH

def windbrain_move_stochastic(used_dirs, board, model, temperature=4.0):
    board_tensor = board_state_to_tensor(used_dirs, board, device=torch.device("cpu"))
    with torch.no_grad():
        q_values = model(board_tensor)
    ideal_direction = torch.argmax(q_values).item()

    [chosen_direction, picked_best] = select_action_with_temperature(q_values, temperature)

    stochastic_logging = False
    if stochastic_logging:
        print(f"Wind Move Q values: {q_values}")
        if picked_best:
            wind_moves_cnt['best'] += 1
            wind_moves_tally.append('_')
        else:
            wind_moves_cnt['rest'] += 1
            wind_moves_tally.append('X')

        if chosen_direction != ideal_direction:
            print(f"Wind Best Move: {ideal_direction} selected: {chosen_direction} {temperature=} {picked_best=}")

    print(f"\n------------------     Wind Move: {chosen_direction}")

    #dir_tuple = dir_pairs[direction_names.index(chosen_direction)]
    dir_tuple = dir_pairs[chosen_direction]
    used_dirs[chosen_direction] = 1
    return dir_tuple

def load_model(model_dir, model_filename="seedbrain.pth", device='cpu'): 
    model_path = os.path.join(model_dir, model_filename)

    checkpoint = torch.load(model_path)
    input_size = checkpoint['input_size']
    hidden_sizes = checkpoint['hidden_sizes']
    output_size = checkpoint['output_size']

    #print(f"Loading model from {model_path}\n {input_size=} {hidden_sizes=} {output_size=}")
    #quit()

    model = DQN(input_size, hidden_sizes, output_size, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

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

# save_parameters not in use anymore?
def save_parameters(subdir, subdir_num, params):
    params_filename = f"params{subdir_num:03d}.py"
    params_filepath = os.path.join(subdir, params_filename)
    
    with open(params_filepath, 'w') as f:
        for key, value in params.items():
            f.write(f"{key} = {repr(value)}\n")
    print(f"Parameters saved to {params_filepath}")


def save_model(model, model_subdir, subdir_num, params, model_filename="seedbrain.pth"):
    os.makedirs(model_subdir, exist_ok=True)
    model_save_path = os.path.join(model_subdir, model_filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': params['INPUT_SIZE'],
        #'hidden_sizes': params['HIDDEN_SIZE'],
        'hidden_sizes': params['MIDDLE_LAYERS'],
        'output_size': params['OUTPUT_SIZE']
    }, model_save_path)
    print(f"Model saved to {model_save_path}")

    params_filename = f"params{subdir_num:03d}.py"
    params_filepath = os.path.join(model_subdir, params_filename)
    with open(params_filepath, 'w') as f:
        for key, value in params.items():
            f.write(f"{key} = {repr(value)}\n")
    print(f"Parameters saved to {params_filepath}")

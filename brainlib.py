import torch


from game import BOARD_WIDTH, BOARD_HEIGHT, NUM_DIR

def board_state_to_tensor(direction_list, board, device=None):

    # it's unclear to me if the direction list is the USED or the UNUSED directions
    # Check this and check EVERYwhere it is used to make sure it is used correctly.
    # (it's possible one trained one way and the other trained the other way..
    # which would work for self training but not for self play)



    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # 1st 8 cells are available directions
    # next 25 cells are dandelions
    # next 25 cells are seeds
    direction_tensor = torch.tensor([1 if direction else 0 for direction in direction_list], dtype=torch.float32).to(device)
    if BOARD_HEIGHT == 0:  # dev
        return direction_tensor 
    dandelion_tensor  = torch.tensor([[1 if cell == 1 else 0 for cell in row] for row in board], dtype=torch.float32).to(device)
    seed_tensor       = torch.tensor([[1 if cell == 2 else 0 for cell in row] for row in board], dtype=torch.float32).to(device)
    return torch.cat((direction_tensor, dandelion_tensor.view(-1), seed_tensor.view(-1)))


def board_state_from_tensor(tensor, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

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
    return [direction_list, board]


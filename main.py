#Dandelions and Wind, from the book Math Games with Bad Drawings.                                                                 
# Playable by 2 players who share a keyboard.                                                                                     

import copy
import torch
import torch.nn as nn 



def initialize_board():    
    return [[0 for _ in range(5)] for _ in range(5)]

def display_board_with_labels(board):
    #print(f"{board=}")
    #tsor = board_to_tensor(board)
    #print(f"{tsor=}")
    col_labels = '  A B C D E'
    print(col_labels)
    for i, row in enumerate(board):
        print(f"{i+1} {' '.join('*' if cell == 1 else '.' if cell == 2 else ' ' for cell in row)}")

def place_dandelion(board, row, col):
    board[row][col] = 1

def board_to_tensor(board):
    dandelion_tensor = torch.tensor([[1 if cell == 1 else 0 for cell in row] for row in board]).view(-1)
    seed_tensor = torch.tensor([[1 if cell == 2 else 0 for cell in row] for row in board]).view(-1)
    return torch.cat((dandelion_tensor, seed_tensor))



def spread_seeds(board, direction):
    new_board = copy.deepcopy(board)
    for row in range(5):
        for col in range(5):
            if board[row][col] == 1:
                dx, dy = direction
                new_row, new_col = row + dx, col + dy
                while 0 <= new_row < 5 and 0 <= new_col < 5:
                    if new_board[new_row][new_col] == 0:
                        new_board[new_row][new_col] = 2
                    new_row += dx
                    new_col += dy
    return new_board


def check_dandelion_win(board):  # check after every Dandelion turn                                                               
    return all(cell in [1, 2] for row in board for cell in row)



def convert_user_input(row_str, col_str):
    row = int(row_str) - 1
    col = ord(col_str.upper()) - ord('A')
    return row, col

# Directions (N, S, E, W, NE, NW, SE, SW)                                                                                         
directions = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
direction_names = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']



def validate_row_input(row_str):
    return row_str.isdigit() and 1 <= int(row_str) <= 5


def validate_col_input(col_str):
    return col_str.upper() in ['A', 'B', 'C', 'D', 'E']


def validate_direction_input(direction_str, available_directions):
    return direction_str.upper() in available_directions

def play_game():

    board = initialize_board()
    available_directions = direction_names.copy()

    # Main game loop                                                                                                              
    for turn in range(7):
        # Dandelion's turn                                                                                                        
        print("Dandelion's Turn:")
        display_board_with_labels(board)

        row_str = input('Enter the row (1-5) to place the dandelion: ')
        while not validate_row_input(row_str):
            print("Invalid row. Please enter a number between 1 and 5.")
            row_str = input('Enter the row (1-5) to place the dandelion: ')

        col_str = input('Enter the column (A-E) to place the dandelion: ')
        while not validate_col_input(col_str):
            print("Invalid column. Please enter a letter between A and E.")
            col_str = input('Enter the column (A-E) to place the dandelion: ')

        row, col = convert_user_input(row_str, col_str)
        place_dandelion(board, row, col)

        # Check for immediate win condition                                                                                       
        if check_dandelion_win(board):
            print('Dandelions win!')
            break

        # Wind's turn                                                                                                             
        print("Wind's Turn:")
        display_board_with_labels(board)
        print('Available directions:', available_directions)

        chosen_direction = input('Choose a direction to blow the wind: ')
        while not validate_direction_input(chosen_direction, available_directions):
            print("Invalid or unavailable direction. Please choose again.")
            chosen_direction = input('Choose a direction to blow the wind: ')

        available_directions.remove(chosen_direction.upper())
        board = spread_seeds(board, directions[direction_names.index(chosen_direction.upper())])

    # Check for win condition if the game wasn't already won                                                                      
    if not check_dandelion_win(board):
        print('Wind wins!')


class NeuralNetwork(nn.Module):

    # might add more layers... Just get prototype working first

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
        )
        '''
        layers = OrderedDict()
        for i in range(len(NN_SIZE) - 1):
            layers[f"layer_{i}"] = nn.Linear(NN_SIZE[i], NN_SIZE[i+1])
            if i < len(NN_SIZE) - 2:  # No ReLU after last layer
                layers[f"relu_{i}"] = nn.ReLU()
        self.linear_relu_stack = nn.Sequential(layers)
        '''

    def forward(self, x):
        logits = self.linear_relu_stack(x)        
        return logits


play_game()

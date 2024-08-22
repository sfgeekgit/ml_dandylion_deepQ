#Dandelions and Wind, from the book Math Games with Bad Drawings.                                                                 
# Playable by 2 players who share a keyboard.                                                                                     

import copy
#import torch
#import torch.nn as nn 


BOARD_HEIGHT = BOARD_WIDTH = 5  # will always be 5
NUM_DIR = 8 # number of directions

#LEARNING_RATE = 0.001
#INPUT_SIZE = BOARD_WIDTH * BOARD_HEIGHT * 2
#HIDDEN_SIZE = 100
#OUTPUT_SIZE = 8

def initialize_board():    
    return [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]

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



def spread_seeds(board, direction_tuple):
    new_board = copy.deepcopy(board)
    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            if board[row][col] == 1:
                dx, dy = direction_tuple
                new_row, new_col = row + dx, col + dy
                while 0 <= new_row < BOARD_HEIGHT and 0 <= new_col < BOARD_WIDTH:
                    if new_board[new_row][new_col] == 0:
                        new_board[new_row][new_col] = 2
                    new_row += dx
                    new_col += dy
    return new_board

def check_dandelion_win(board):  # check after every Dandelion turn                                                               
    # return all(cell in [1, 2] for row in board for cell in row)
    # ^ above works fine, but checks every cell.
    # more efficent to return early on any zero
    for row in board:
        for cell in row:
            if cell == 0:
                return False
    return True


def convert_user_input(row_str, col_str):
    row = int(row_str) - 1
    col = ord(col_str.upper()) - ord('A')
    return row, col

# Directions (N, S, E, W, NE, NW, SE, SW)                                                                                         
dir_pairs = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
direction_names = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']

def validate_row_input(row_str):
    return row_str.isdigit() and 1 <= int(row_str) <= BOARD_HEIGHT

def validate_col_input(col_str):
    return col_str.upper() in [chr(i + ord('A')) for i in range(BOARD_WIDTH)]

def validate_direction_input(direction_str, available_directions):
    return direction_str.upper() in available_directions


def get_human_wind_move(board, used_directions): 
    available_directions = [direction_names[i] for i in range(len(direction_names)) if not used_directions[i]]

    print("Wind's Turn:")
    display_board_with_labels(board)
    print('Available directions:', available_directions)

    chosen_direction = input('Choose a direction to blow the wind: ').upper()
    while not validate_direction_input(chosen_direction, available_directions):
        print("Invalid or unavailable direction. Please choose again.")
        chosen_direction = input('Choose a direction to blow the wind: ').upper()

    dir_tuple = dir_pairs[direction_names.index(chosen_direction)]
    used_directions[direction_names.index(chosen_direction)] = 1
    return dir_tuple


def play_game():
    board = initialize_board()
    available_directions = direction_names.copy()
    used_directions = [0] * NUM_DIR 


    # Main game loop                                                                                                              
    for turn in range(7):
        # Dandelion's turn                                                                                                        
        print("Dandelion's Turn:")
        display_board_with_labels(board)

        row_str = input('Enter the row (1-{}) to place the dandelion: '.format(BOARD_HEIGHT))
        while not validate_row_input(row_str):
            print("Invalid row. Please enter a number between 1 and {}.".format(BOARD_HEIGHT))
            row_str = input('Enter the row (1-{}) to place the dandelion: '.format(BOARD_HEIGHT))

        col_str = input('Enter the column (A-{}) to place the dandelion: '.format(chr(BOARD_WIDTH + ord('A') - 1)))
        while not validate_col_input(col_str):
            print("Invalid column. Please enter a letter between A and {}.".format(chr(BOARD_WIDTH + ord('A') - 1)))
            col_str = input('Enter the column (A-{}) to place the dandelion: '.format(chr(BOARD_WIDTH + ord('A') - 1)))

        row, col = convert_user_input(row_str, col_str)
        place_dandelion(board, row, col)

        # Check for immediate win condition                                                                                       
        if check_dandelion_win(board):
            print('Dandelions win!')
            break


        dir_tuple = get_human_wind_move(board, used_directions)
        board = spread_seeds(board, dir_tuple)

    # Check for win condition if the game wasn't already won                                                                      
    if not check_dandelion_win(board):
        print('Wind wins!')



#####################
#       Main        #

if __name__ == "__main__":
    play_game()
    #train_nn()

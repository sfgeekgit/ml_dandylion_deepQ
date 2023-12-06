#Dandelions and Wind, from the book Math Games with Bad Drawings.                                                                 
# Playable by 2 players who share a keyboard.                                                                                     

import copy

def initialize_board():
    return [[' ' for _ in range(5)] for _ in range(5)]


def display_board_with_labels(board):
    col_labels = '  A B C D E'
    print(col_labels)
    for i, row in enumerate(board):
        print(f"{i+1} {' '.join(row)}")

def place_dandelion(board, row, col):
    board[row][col] = '*'


def spread_seeds(board, direction):
    new_board = copy.deepcopy(board)
    for row in range(5):
        for col in range(5):
            if board[row][col] == '*':
                dx, dy = direction
                new_row, new_col = row + dx, col + dy
                while 0 <= new_row < 5 and 0 <= new_col < 5:
                    if new_board[new_row][new_col] == ' ':
                        new_board[new_row][new_col] = '.'
                    new_row += dx
                    new_col += dy
    return new_board


def check_dandelion_win(board):  # check after every Dandelion turn                                                               
    return all(cell in ['*', '.'] for row in board for cell in row)



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

play_game()

"""
Dandelion Game - Phase 2: Web App for 2-Player Game
A Flask web app for the Dandelion game from "Math Games with Bad Drawings"
"""

from flask import Flask, render_template, jsonify, request, session
import copy
import os

from game import (
    initialize_board, place_dandelion, spread_seeds, check_dandelion_win,
    dir_pairs, direction_names, BOARD_HEIGHT, BOARD_WIDTH, NUM_DIR
)

app = Flask(__name__)
app.secret_key = os.urandom(24)

NUM_TURNS = 7

def create_new_game():
    """Initialize a new game state."""
    return {
        'board': initialize_board(),
        'used_directions': [0] * NUM_DIR,
        'turn': 0,
        'phase': 'dandelion',  # 'dandelion' or 'wind'
        'winner': None,
        'game_over': False
    }

def get_available_directions(used_directions):
    """Get list of available direction names."""
    return [direction_names[i] for i in range(len(direction_names)) if not used_directions[i]]

def get_game_state():
    """Get current game state from session."""
    if 'game' not in session:
        session['game'] = create_new_game()
    return session['game']

def save_game_state(game):
    """Save game state to session."""
    session['game'] = game
    session.modified = True

def add_stats(game):
    """Add computed stats (flowers, seeds, empty) to game state for API response."""
    board = game['board']
    game['flowers'] = sum(cell == 1 for row in board for cell in row)
    game['seeds'] = sum(cell == 2 for row in board for cell in row)
    game['empty'] = sum(cell == 0 for row in board for cell in row)
    return game

@app.route('/')
def index():
    # Generate column labels (A, B, C, ...)
    col_labels = [chr(ord('A') + i) for i in range(BOARD_WIDTH)]
    row_labels = [str(i + 1) for i in range(BOARD_HEIGHT)]
    # Clockwise order for compass display
    directions_clockwise = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    return render_template(
        'game.html',
        board_width=BOARD_WIDTH,
        board_height=BOARD_HEIGHT,
        num_turns=NUM_TURNS,
        col_labels=col_labels,
        row_labels=row_labels,
        directions_clockwise=directions_clockwise,
        direction_names=direction_names
    )

@app.route('/api/state')
def get_state():
    return jsonify(add_stats(get_game_state()))

@app.route('/api/new', methods=['POST'])
def new_game():
    game = create_new_game()
    save_game_state(game)
    return jsonify(add_stats(game))

@app.route('/api/place', methods=['POST'])
def place():
    data = request.json
    row, col = data['row'], data['col']

    game = get_game_state()

    if game['game_over'] or game['phase'] != 'dandelion':
        return jsonify(add_stats(game))

    # Place dandelion using game.py function
    board = copy.deepcopy(game['board'])
    place_dandelion(board, row, col)
    game['board'] = board

    # Check for win using game.py function
    if check_dandelion_win(game['board']):
        game['winner'] = 'dandelion'
        game['game_over'] = True
    else:
        game['phase'] = 'wind'

    save_game_state(game)
    return jsonify(add_stats(game))

@app.route('/api/wind', methods=['POST'])
def wind():
    data = request.json
    direction = data['direction']

    game = get_game_state()

    if game['game_over'] or game['phase'] != 'wind':
        return jsonify(add_stats(game))

    # Check if direction already used
    dir_index = direction_names.index(direction)
    if game['used_directions'][dir_index] == 1:
        return jsonify(add_stats(game))

    # Spread seeds using game.py function
    dir_tuple = dir_pairs[dir_index]
    game['board'] = spread_seeds(game['board'], dir_tuple)
    game['used_directions'][dir_index] = 1

    # Check for win after wind
    if check_dandelion_win(game['board']):
        game['winner'] = 'dandelion'
        game['game_over'] = True
    else:
        game['turn'] += 1
        if game['turn'] >= NUM_TURNS:
            game['winner'] = 'wind'
            game['game_over'] = True
        else:
            game['phase'] = 'dandelion'

    save_game_state(game)
    return jsonify(add_stats(game))

if __name__ == '__main__':
    app.run(port=5002, debug=True)

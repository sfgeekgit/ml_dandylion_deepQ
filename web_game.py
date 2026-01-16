"""
Dandelion Game - Web Interface
A Flask web app for the Dandelion game from "Math Games with Bad Drawings"
Supports Human vs Human, Human vs AI, and AI vs AI modes.
"""

from flask import Flask, render_template, jsonify, request, session
import copy
import os

from game import (
    initialize_board, place_dandelion, spread_seeds, check_dandelion_win,
    dir_pairs, direction_names, BOARD_HEIGHT, BOARD_WIDTH, NUM_DIR
)
from brainlib import load_model, seedbrain_move_stochastic, windbrain_move_stochastic
from model_utils import (
    discover_models, load_params, format_params_for_display,
    models_available, seed_models_available, wind_models_available
)

app = Flask(__name__)
app.secret_key = os.urandom(24)

NUM_TURNS = 7

# Model cache - loaded models stored at app level to avoid reloading
loaded_models = {}


def get_or_load_model(model_type, model_id):
    """Get model from cache or load it."""
    cache_key = f"{model_type}/{model_id}"
    if cache_key not in loaded_models:
        model_dir = f"models/{model_type}/{model_id}"
        model_file = "seedbrain.pth" if model_type == "seeds" else "windbrain.pth"
        loaded_models[cache_key] = load_model(model_dir, model_file)
    return loaded_models[cache_key]


def create_new_game(game_mode='hvh', ai_config=None):
    """
    Initialize a new game state.

    Args:
        game_mode: 'hvh' (human vs human), 'hvai' (human vs ai), 'aivsai' (ai vs ai)
        ai_config: Dict with AI configuration
            For hvai: {'human_role': 'dandelion'|'wind', 'model_type': str, 'model_id': str, 'ai_temp': float}
            For aivsai: {'seed_model_id': str, 'wind_model_id': str, 'seed_temp': float, 'wind_temp': float}
    """
    return {
        'board': initialize_board(),
        'used_directions': [0] * NUM_DIR,
        'turn': 0,
        'phase': 'dandelion',  # 'dandelion' or 'wind'
        'winner': None,
        'game_over': False,
        'game_mode': game_mode,
        'ai_config': ai_config or {}
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

def get_game_template_vars(game_mode='hvh'):
    """Get common template variables for game pages."""
    col_labels = [chr(ord('A') + i) for i in range(BOARD_WIDTH)]
    row_labels = [str(i + 1) for i in range(BOARD_HEIGHT)]
    directions_clockwise = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    return {
        'board_width': BOARD_WIDTH,
        'board_height': BOARD_HEIGHT,
        'num_turns': NUM_TURNS,
        'col_labels': col_labels,
        'row_labels': row_labels,
        'directions_clockwise': directions_clockwise,
        'direction_names': direction_names,
        'game_mode': game_mode
    }


@app.route('/')
def landing():
    """Landing page with game mode selection."""
    seed_models = discover_models('seeds')
    wind_models = discover_models('wind')
    return render_template(
        'index.html',
        has_seed_models=len(seed_models) > 0,
        has_wind_models=len(wind_models) > 0,
        seed_count=len(seed_models),
        wind_count=len(wind_models)
    )


@app.route('/play/hvh')
def play_hvh():
    """Human vs Human game."""
    return render_template('game.html', **get_game_template_vars('hvh'))


@app.route('/play/hvai')
def setup_hvai():
    """Setup page for Human vs AI."""
    seed_models = discover_models('seeds')
    wind_models = discover_models('wind')
    if not seed_models and not wind_models:
        return render_template('no_models.html')

    # Load params for each model
    seed_models_with_params = []
    for m in seed_models:
        params = load_params('seeds', m['id'])
        m['params'] = format_params_for_display(params) if params else {}
        seed_models_with_params.append(m)

    wind_models_with_params = []
    for m in wind_models:
        params = load_params('wind', m['id'])
        m['params'] = format_params_for_display(params) if params else {}
        wind_models_with_params.append(m)

    return render_template(
        'setup.html',
        mode='hvai',
        seed_models=seed_models_with_params,
        wind_models=wind_models_with_params
    )


@app.route('/play/hvai/game')
def play_hvai():
    """Human vs AI game page."""
    return render_template('game.html', **get_game_template_vars('hvai'))


@app.route('/play/aivsai')
def setup_aivsai():
    """Setup page for AI vs AI."""
    seed_models = discover_models('seeds')
    wind_models = discover_models('wind')
    if not seed_models or not wind_models:
        return render_template(
            'no_models.html',
            message="Need at least one seed AND one wind model for AI vs AI"
        )

    # Load params for each model
    for m in seed_models:
        params = load_params('seeds', m['id'])
        m['params'] = format_params_for_display(params) if params else {}

    for m in wind_models:
        params = load_params('wind', m['id'])
        m['params'] = format_params_for_display(params) if params else {}

    return render_template(
        'setup.html',
        mode='aivsai',
        seed_models=seed_models,
        wind_models=wind_models
    )


@app.route('/play/aivsai/game')
def play_aivsai():
    """AI vs AI spectator game page."""
    return render_template('game.html', **get_game_template_vars('aivsai'))

@app.route('/api/state')
def get_state():
    return jsonify(add_stats(get_game_state()))

@app.route('/api/new', methods=['POST'])
def new_game():
    data = request.json or {}
    game_mode = data.get('game_mode', 'hvh')
    ai_config = data.get('ai_config')

    game = create_new_game(game_mode, ai_config)
    save_game_state(game)
    return jsonify(add_stats(game))


@app.route('/api/models/<model_type>')
def list_models(model_type):
    """List available models of a given type."""
    if model_type not in ['seeds', 'wind']:
        return jsonify({'error': 'Invalid model type'}), 400
    models = discover_models(model_type)
    return jsonify({'models': models})


@app.route('/api/models/<model_type>/<model_id>/params')
def get_model_params(model_type, model_id):
    """Get hyperparameters for a specific model."""
    if model_type not in ['seeds', 'wind']:
        return jsonify({'error': 'Invalid model type'}), 400
    params = load_params(model_type, model_id)
    if not params:
        return jsonify({'error': 'Model not found'}), 404
    display = format_params_for_display(params)
    return jsonify(display)


@app.route('/api/ai-move', methods=['POST'])
def ai_move():
    """Execute AI move for current phase."""
    game = get_game_state()

    if game['game_over']:
        return jsonify(add_stats(game))

    game_mode = game.get('game_mode', 'hvh')
    if game_mode not in ['hvai', 'aivsai']:
        return jsonify({'error': 'Not an AI game'}), 400

    ai_config = game.get('ai_config', {})
    current_phase = game['phase']

    # For hvai mode, check if it's actually the AI's turn
    if game_mode == 'hvai':
        human_role = ai_config.get('human_role', 'dandelion')
        if current_phase == human_role:
            return jsonify({'error': 'Human turn, not AI'}), 400

    board = copy.deepcopy(game['board'])
    used_dirs = game['used_directions'][:]

    if current_phase == 'dandelion':
        # AI plays as dandelion (seed brain)
        if game_mode == 'aivsai':
            model_id = ai_config.get('seed_model_id')
            temp = ai_config.get('seed_temp', 0.0)
        else:  # hvai, AI is playing dandelion
            model_id = ai_config.get('model_id')
            temp = ai_config.get('ai_temp', 0.0)

        model = get_or_load_model('seeds', model_id)
        row, col = seedbrain_move_stochastic(used_dirs, board, model, temp)

        # Check for illegal move (placing on existing dandelion)
        if board[row][col] == 1:
            # AI made illegal move - forfeit
            game['winner'] = 'wind'
            game['game_over'] = True
            game['forfeit'] = 'dandelion'
        else:
            place_dandelion(board, row, col)
            game['board'] = board

            if check_dandelion_win(game['board']):
                game['winner'] = 'dandelion'
                game['game_over'] = True
            else:
                game['phase'] = 'wind'

    else:  # wind phase
        # AI plays as wind
        if game_mode == 'aivsai':
            model_id = ai_config.get('wind_model_id')
            temp = ai_config.get('wind_temp', 0.0)
        else:  # hvai, AI is playing wind
            model_id = ai_config.get('model_id')
            temp = ai_config.get('ai_temp', 0.0)

        model = get_or_load_model('wind', model_id)

        # Store used_dirs before move to detect illegal reuse
        used_dirs_before = used_dirs[:]
        dir_tuple = windbrain_move_stochastic(used_dirs, board, model, temp)

        # Find which direction was chosen
        dir_index = dir_pairs.index(dir_tuple)

        # Check for illegal move (reusing direction)
        if used_dirs_before[dir_index] == 1:
            # AI made illegal move - forfeit
            game['winner'] = 'dandelion'
            game['game_over'] = True
            game['forfeit'] = 'wind'
        else:
            game['board'] = spread_seeds(game['board'], dir_tuple)
            game['used_directions'][dir_index] = 1

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

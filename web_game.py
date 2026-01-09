"""
Dandelion Game - Phase 2: Web App for 2-Player Game
A Flask web app for the Dandelion game from "Math Games with Bad Drawings"
"""

from flask import Flask, render_template_string, jsonify, request, session
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


# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dandelions - A Game of Seeds and Wind</title>
    <link href="https://fonts.googleapis.com/css2?family=Patrick+Hand&family=Nunito:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --meadow-light: #e8f5e9;
            --meadow: #c8e6c9;
            --meadow-dark: #a5d6a7;
            --dandelion-yellow: #fdd835;
            --dandelion-green: #43a047;
            --seed-green: #66bb6a;
            --wind-blue: #64b5f6;
            --wind-dark: #1976d2;
            --sky: #e3f2fd;
            --earth: #8d6e63;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            min-height: 100vh;
            background: linear-gradient(180deg, var(--sky) 0%, var(--meadow-light) 50%, var(--meadow) 100%);
            font-family: 'Nunito', sans-serif;
            padding: 20px;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
        }

        h1 {
            font-family: 'Patrick Hand', cursive;
            font-size: 3rem;
            color: var(--dandelion-green);
            text-align: center;
            margin-bottom: 10px;
            text-shadow: 2px 2px 0 white;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-style: italic;
        }

        .game-area {
            display: flex;
            justify-content: center;
            align-items: flex-start;
            gap: 40px;
            flex-wrap: wrap;
        }

        .board-container {
            background: white;
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }

        .board {
            display: grid;
            grid-template-columns: repeat({{ board_width }}, 70px);
            grid-template-rows: repeat({{ board_height }}, 70px);
            gap: 4px;
            background: var(--earth);
            padding: 4px;
            border-radius: 10px;
        }

        .cell {
            background: var(--meadow-light);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
        }

        .cell:hover:not(.occupied) {
            background: var(--meadow);
            transform: scale(1.05);
        }

        .cell.dandelion {
            background: var(--meadow);
        }

        .cell.seed {
            background: var(--meadow-light);
        }

        .cell .dandelion-icon {
            color: var(--dandelion-green);
            font-size: 2.8rem;
            filter: drop-shadow(1px 1px 1px rgba(0,0,0,0.2));
        }

        .cell .seed-icon {
            width: 16px;
            height: 16px;
            background: var(--seed-green);
            border-radius: 50%;
            box-shadow: inset -2px -2px 4px rgba(0,0,0,0.2);
        }

        .col-labels, .row-labels {
            display: flex;
            font-family: 'Patrick Hand', cursive;
            font-size: 1.2rem;
            color: var(--earth);
        }

        .col-labels {
            justify-content: space-around;
            margin-bottom: 8px;
            padding: 0 4px;
        }

        .col-labels span {
            width: 70px;
            text-align: center;
        }

        .board-with-labels {
            display: flex;
        }

        .row-labels {
            flex-direction: column;
            justify-content: space-around;
            margin-right: 8px;
            padding: 4px 0;
        }

        .row-labels span {
            height: 70px;
            display: flex;
            align-items: center;
        }

        .compass-container {
            background: white;
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            text-align: center;
        }

        .compass-title {
            font-family: 'Patrick Hand', cursive;
            font-size: 1.5rem;
            color: var(--wind-dark);
            margin-bottom: 15px;
        }

        .compass {
            width: 200px;
            height: 200px;
            position: relative;
            margin: 0 auto;
        }

        .compass-bg {
            width: 100%;
            height: 100%;
            border: 3px solid var(--wind-blue);
            border-radius: 50%;
            background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
        }

        .direction-btn {
            position: absolute;
            width: 44px;
            height: 44px;
            border-radius: 50%;
            border: 2px solid var(--wind-dark);
            background: var(--wind-blue);
            color: white;
            font-family: 'Patrick Hand', cursive;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .direction-btn:hover:not(:disabled) {
            transform: scale(1.15);
            background: var(--wind-dark);
        }

        .direction-btn:disabled {
            background: #ccc;
            border-color: #999;
            cursor: not-allowed;
            opacity: 0.5;
        }

        .direction-btn.used {
            background: #999;
            border-color: #666;
            text-decoration: line-through;
        }

        /* Position directions around compass */
        .dir-N  { top: -5px; left: 50%; transform: translateX(-50%); }
        .dir-NE { top: 15px; right: 15px; }
        .dir-E  { top: 50%; right: -5px; transform: translateY(-50%); }
        .dir-SE { bottom: 15px; right: 15px; }
        .dir-S  { bottom: -5px; left: 50%; transform: translateX(-50%); }
        .dir-SW { bottom: 15px; left: 15px; }
        .dir-W  { top: 50%; left: -5px; transform: translateY(-50%); }
        .dir-NW { top: 15px; left: 15px; }

        .status-panel {
            background: white;
            border-radius: 15px;
            padding: 20px 30px;
            margin-top: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            text-align: center;
        }

        .turn-indicator {
            font-family: 'Patrick Hand', cursive;
            font-size: 1.8rem;
            margin-bottom: 10px;
        }

        .turn-indicator.dandelion-turn {
            color: var(--dandelion-green);
        }

        .turn-indicator.wind-turn {
            color: var(--wind-dark);
        }

        .turn-count {
            color: #666;
            font-size: 1rem;
        }

        .winner-announcement {
            font-family: 'Patrick Hand', cursive;
            font-size: 2.5rem;
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
            animation: celebrate 0.5s ease;
        }

        .winner-announcement.dandelion-wins {
            background: linear-gradient(135deg, var(--meadow-light), var(--meadow));
            color: var(--dandelion-green);
        }

        .winner-announcement.wind-wins {
            background: linear-gradient(135deg, var(--sky), var(--wind-blue));
            color: var(--wind-dark);
        }

        @keyframes celebrate {
            0% { transform: scale(0.8); opacity: 0; }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); opacity: 1; }
        }

        .new-game-btn {
            background: var(--dandelion-green);
            color: white;
            border: none;
            padding: 12px 30px;
            font-family: 'Patrick Hand', cursive;
            font-size: 1.3rem;
            border-radius: 25px;
            cursor: pointer;
            margin-top: 15px;
            transition: all 0.2s ease;
        }

        .new-game-btn:hover {
            background: #2e7d32;
            transform: scale(1.05);
        }

        .rules-panel {
            background: rgba(255,255,255,0.9);
            border-radius: 15px;
            padding: 20px;
            margin-top: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }

        .rules-panel h3 {
            font-family: 'Patrick Hand', cursive;
            color: var(--dandelion-green);
            margin-bottom: 10px;
        }

        .rules-panel ul {
            margin-left: 20px;
            color: #555;
        }

        .rules-panel li {
            margin-bottom: 8px;
        }

        .legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 15px;
            flex-wrap: wrap;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9rem;
            color: #666;
        }

        .legend-dandelion {
            font-size: 1.5rem;
            color: var(--dandelion-green);
        }

        .legend-seed {
            width: 12px;
            height: 12px;
            background: var(--seed-green);
            border-radius: 50%;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dandelions</h1>
        <p class="subtitle">A game of seeds and wind from "Math Games with Bad Drawings"</p>

        <div class="game-area">
            <div class="board-container">
                <div class="col-labels">
                    {% for col in col_labels %}<span>{{ col }}</span>{% endfor %}
                </div>
                <div class="board-with-labels">
                    <div class="row-labels">
                        {% for row in row_labels %}<span>{{ row }}</span>{% endfor %}
                    </div>
                    <div class="board" id="board">
                        <!-- Cells generated by JavaScript -->
                    </div>
                </div>
                <div class="legend">
                    <div class="legend-item">
                        <span class="legend-dandelion">*</span>
                        <span>Dandelion</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-seed"></span>
                        <span>Seed</span>
                    </div>
                </div>
            </div>

            <div class="compass-container">
                <div class="compass-title">Wind Direction</div>
                <div class="compass">
                    <div class="compass-bg"></div>
                    {% for dir in directions_clockwise %}
                    <button class="direction-btn dir-{{ dir }}" data-dir="{{ dir }}">{{ dir }}</button>
                    {% endfor %}
                </div>
            </div>
        </div>

        <div class="status-panel">
            <div class="turn-indicator" id="turnIndicator">Dandelion's Turn: Place a flower!</div>
            <div class="turn-count" id="turnCount">Turn 1 of {{ num_turns }}</div>
            <div id="winnerAnnouncement"></div>
            <button class="new-game-btn" id="newGameBtn">New Game</button>
        </div>

        <div class="rules-panel">
            <h3>How to Play</h3>
            <ul>
                <li><strong>Dandelion</strong> places a flower (*) anywhere on the grid by clicking a cell.</li>
                <li><strong>Wind</strong> blows in one of 8 directions by clicking a compass button.</li>
                <li>When wind blows, seeds spread from ALL dandelions in that direction.</li>
                <li>Each wind direction can only be used once!</li>
                <li>After {{ num_turns }} turns each: <strong>Dandelions win</strong> if the board is full, otherwise <strong>Wind wins</strong>!</li>
            </ul>
        </div>
    </div>

    <script>
        const BOARD_HEIGHT = {{ board_height }};
        const BOARD_WIDTH = {{ board_width }};
        const NUM_TURNS = {{ num_turns }};
        const DIRECTION_NAMES = {{ direction_names | tojson }};

        let gameState = null;

        async function fetchGameState() {
            const response = await fetch('/api/state');
            gameState = await response.json();
            renderGame();
        }

        async function newGame() {
            const response = await fetch('/api/new', { method: 'POST' });
            gameState = await response.json();
            renderGame();
        }

        async function placeDandelion(row, col) {
            if (gameState.phase !== 'dandelion' || gameState.game_over) return;
            if (gameState.board[row][col] === 1) return; // Can't place on existing dandelion

            const response = await fetch('/api/place', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ row, col })
            });
            gameState = await response.json();
            renderGame();
        }

        async function blowWind(direction) {
            if (gameState.phase !== 'wind' || gameState.game_over) return;

            // Check if direction is used (used_directions is an array of 0/1)
            const dirIndex = DIRECTION_NAMES.indexOf(direction);
            if (gameState.used_directions[dirIndex] === 1) return;

            const response = await fetch('/api/wind', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ direction })
            });
            gameState = await response.json();
            renderGame();
        }

        function renderGame() {
            const boardEl = document.getElementById('board');
            const turnIndicator = document.getElementById('turnIndicator');
            const turnCount = document.getElementById('turnCount');
            const winnerAnnouncement = document.getElementById('winnerAnnouncement');

            // Render board
            boardEl.innerHTML = '';
            for (let row = 0; row < BOARD_HEIGHT; row++) {
                for (let col = 0; col < BOARD_WIDTH; col++) {
                    const cell = document.createElement('div');
                    cell.className = 'cell';
                    const value = gameState.board[row][col];

                    if (value === 1) {
                        cell.classList.add('dandelion', 'occupied');
                        cell.innerHTML = '<span class="dandelion-icon">*</span>';
                    } else if (value === 2) {
                        cell.classList.add('seed', 'occupied');
                        cell.innerHTML = '<span class="seed-icon"></span>';
                    }

                    if (gameState.phase === 'dandelion' && !gameState.game_over && value !== 1) {
                        cell.addEventListener('click', () => placeDandelion(row, col));
                    }

                    boardEl.appendChild(cell);
                }
            }

            // Render compass buttons
            document.querySelectorAll('.direction-btn').forEach(btn => {
                const dir = btn.dataset.dir;
                const dirIndex = DIRECTION_NAMES.indexOf(dir);
                const isUsed = gameState.used_directions[dirIndex] === 1;
                btn.disabled = isUsed || gameState.phase !== 'wind' || gameState.game_over;
                btn.classList.toggle('used', isUsed);
            });

            // Update turn indicator
            if (gameState.game_over) {
                if (gameState.winner === 'dandelion') {
                    turnIndicator.textContent = 'Game Over!';
                    turnIndicator.className = 'turn-indicator dandelion-turn';
                    winnerAnnouncement.innerHTML = '<div class="winner-announcement dandelion-wins">The Dandelions Win! The meadow is covered!</div>';
                } else {
                    turnIndicator.textContent = 'Game Over!';
                    turnIndicator.className = 'turn-indicator wind-turn';
                    winnerAnnouncement.innerHTML = '<div class="winner-announcement wind-wins">The Wind Wins! Empty spots remain!</div>';
                }
            } else {
                winnerAnnouncement.innerHTML = '';
                if (gameState.phase === 'dandelion') {
                    turnIndicator.textContent = "Dandelion's Turn: Place a flower!";
                    turnIndicator.className = 'turn-indicator dandelion-turn';
                } else {
                    turnIndicator.textContent = "Wind's Turn: Choose a direction!";
                    turnIndicator.className = 'turn-indicator wind-turn';
                }
            }

            turnCount.textContent = `Turn ${gameState.turn + 1} of ${NUM_TURNS}`;
        }

        // Event listeners
        document.getElementById('newGameBtn').addEventListener('click', newGame);
        document.querySelectorAll('.direction-btn').forEach(btn => {
            btn.addEventListener('click', () => blowWind(btn.dataset.dir));
        });

        // Initialize
        fetchGameState();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    # Generate column labels (A, B, C, ...)
    col_labels = [chr(ord('A') + i) for i in range(BOARD_WIDTH)]
    row_labels = [str(i + 1) for i in range(BOARD_HEIGHT)]
    # Clockwise order for compass display
    directions_clockwise = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    return render_template_string(
        HTML_TEMPLATE,
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
    return jsonify(get_game_state())

@app.route('/api/new', methods=['POST'])
def new_game():
    game = create_new_game()
    save_game_state(game)
    return jsonify(game)

@app.route('/api/place', methods=['POST'])
def place():
    data = request.json
    row, col = data['row'], data['col']

    game = get_game_state()

    if game['game_over'] or game['phase'] != 'dandelion':
        return jsonify(game)

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
    return jsonify(game)

@app.route('/api/wind', methods=['POST'])
def wind():
    data = request.json
    direction = data['direction']

    game = get_game_state()

    if game['game_over'] or game['phase'] != 'wind':
        return jsonify(game)

    # Check if direction already used
    dir_index = direction_names.index(direction)
    if game['used_directions'][dir_index] == 1:
        return jsonify(game)

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
    return jsonify(game)

if __name__ == '__main__':
    app.run(port=5002, debug=True)

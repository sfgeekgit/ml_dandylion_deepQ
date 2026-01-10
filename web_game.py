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

def add_stats(game):
    """Add computed stats (flowers, seeds, empty) to game state for API response."""
    board = game['board']
    game['flowers'] = sum(cell == 1 for row in board for cell in row)
    game['seeds'] = sum(cell == 2 for row in board for cell in row)
    game['empty'] = sum(cell == 0 for row in board for cell in row)
    return game


# HTML Template - Dark Dashboard Theme
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dandelions - A Game of Seeds and Wind</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            min-height: 100vh;
            background: #0f0f0f;
            font-family: 'Space Grotesk', sans-serif;
            color: #e0e0e0;
            padding: 30px;
        }

        .dashboard {
            max-width: 1000px;
            margin: 0 auto;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #2a2a2a;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #4ade80, #22c55e);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }

        h1 {
            font-size: 1.5rem;
            font-weight: 600;
            color: white;
        }

        .header-stats {
            display: flex;
            gap: 30px;
        }

        .stat {
            text-align: center;
        }

        .stat-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem;
            font-weight: 600;
            color: white;
        }

        .stat-label {
            font-size: 0.75rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 280px;
            gap: 25px;
        }

        .panel {
            background: #1a1a1a;
            border-radius: 16px;
            border: 1px solid #2a2a2a;
            overflow: hidden;
        }

        .panel-header {
            padding: 15px 20px;
            border-bottom: 1px solid #2a2a2a;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .panel-title {
            font-size: 0.85rem;
            font-weight: 500;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .panel-badge {
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .panel-badge.active {
            background: rgba(74, 222, 128, 0.15);
            color: #4ade80;
        }

        .panel-badge.waiting {
            background: rgba(96, 165, 250, 0.15);
            color: #60a5fa;
        }

        .panel-content {
            padding: 20px;
        }

        .board-container {
            display: flex;
            justify-content: center;
        }

        .board-grid {
            display: grid;
            grid-template-columns: 30px repeat({{ board_width }}, 70px);
            grid-template-rows: 30px repeat({{ board_height }}, 70px);
            gap: 3px;
        }

        .corner {
            /* Empty corner cell */
        }

        .col-label, .row-label {
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #555;
        }

        .cell {
            background: #252525;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s ease;
            border: 1px solid #333;
        }

        .cell.clickable {
            cursor: pointer;
        }

        .cell.clickable:hover {
            background: #303030;
            border-color: #4ade80;
        }

        .cell.dandelion {
            background: rgba(74, 222, 128, 0.15);
            border-color: #4ade80;
        }

        .cell.seed {
            background: rgba(74, 222, 128, 0.08);
            border-color: #333;
        }

        .dandelion-icon {
            font-size: 2rem;
            color: #4ade80;
            text-shadow: 0 0 20px rgba(74, 222, 128, 0.5);
        }

        .seed-icon {
            width: 12px;
            height: 12px;
            background: #4ade80;
            border-radius: 50%;
            opacity: 0.6;
            box-shadow: 0 0 10px rgba(74, 222, 128, 0.3);
        }

        .compass-panel .panel-content {
            padding: 15px;
        }

        .compass-display {
            width: 180px;
            height: 180px;
            margin: 0 auto 20px;
            position: relative;
        }

        .compass-ring {
            width: 100%;
            height: 100%;
            border: 2px solid #333;
            border-radius: 50%;
            background: radial-gradient(circle, #1a1a1a 0%, #0f0f0f 100%);
        }

        .compass-center {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 50px;
            height: 50px;
            background: #252525;
            border-radius: 50%;
            border: 2px solid #333;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }

        .dir-btn {
            position: absolute;
            width: 38px;
            height: 38px;
            border-radius: 8px;
            border: 1px solid #444;
            background: #252525;
            color: #60a5fa;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .dir-btn:hover:not(:disabled) {
            background: #60a5fa;
            color: #0f0f0f;
            border-color: #60a5fa;
            box-shadow: 0 0 20px rgba(96, 165, 250, 0.3);
        }

        .dir-btn:disabled {
            cursor: not-allowed;
        }

        .dir-btn.used {
            background: #1a1a1a;
            border-color: #2a2a2a;
            color: #444;
            text-decoration: line-through;
        }

        .dir-N  { top: 5px; left: 50%; transform: translateX(-50%); }
        .dir-NE { top: 25px; right: 25px; }
        .dir-E  { top: 50%; right: 5px; transform: translateY(-50%); }
        .dir-SE { bottom: 25px; right: 25px; }
        .dir-S  { bottom: 5px; left: 50%; transform: translateX(-50%); }
        .dir-SW { bottom: 25px; left: 25px; }
        .dir-W  { top: 50%; left: 5px; transform: translateY(-50%); }
        .dir-NW { top: 25px; left: 25px; }

        .info-panel {
            margin-top: 25px;
        }

        .turn-display {
            text-align: center;
            padding: 20px;
        }

        .current-turn {
            font-size: 0.75rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .turn-player {
            font-size: 1.25rem;
            font-weight: 600;
            color: #4ade80;
        }

        .turn-player.wind {
            color: #60a5fa;
        }

        .action-buttons {
            display: flex;
            gap: 10px;
            padding: 15px 20px;
            border-top: 1px solid #2a2a2a;
        }

        .btn {
            flex: 1;
            padding: 10px;
            border-radius: 8px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .btn-primary {
            background: #4ade80;
            color: #0f0f0f;
            border: none;
        }

        .btn-primary:hover {
            background: #22c55e;
        }

        .legend-row {
            display: flex;
            justify-content: center;
            gap: 20px;
            padding: 15px;
            border-top: 1px solid #2a2a2a;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.75rem;
            color: #666;
        }

        .legend-icon {
            width: 24px;
            height: 24px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .legend-icon.dandelion {
            background: rgba(74, 222, 128, 0.15);
            color: #4ade80;
        }

        .legend-icon.seed {
            background: rgba(74, 222, 128, 0.08);
        }

        .legend-icon.seed span {
            width: 8px;
            height: 8px;
            background: #4ade80;
            opacity: 0.6;
            border-radius: 50%;
        }

        .rules-panel {
            margin-top: 25px;
        }

        .rules-panel .panel-content {
            padding: 20px 25px;
        }

        .rules-subtitle {
            color: #888;
            font-size: 0.9rem;
            font-style: italic;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #2a2a2a;
        }

        .rules-panel h3 {
            color: #4ade80;
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
        }

        .rules-panel ul {
            list-style: none;
            color: #aaa;
            font-size: 0.9rem;
            line-height: 1.8;
        }

        .rules-panel li {
            padding-left: 20px;
            position: relative;
            margin-bottom: 8px;
        }

        .rules-panel li::before {
            content: '>';
            position: absolute;
            left: 0;
            color: #4ade80;
            font-family: 'JetBrains Mono', monospace;
        }

        .rules-panel strong {
            color: #e0e0e0;
        }

        .winner-announcement {
            font-size: 1.5rem;
            font-weight: 600;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
            text-align: center;
            animation: celebrate 0.5s ease;
        }

        .winner-announcement.dandelion-wins {
            background: rgba(74, 222, 128, 0.15);
            color: #4ade80;
            border: 1px solid #4ade80;
        }

        .winner-announcement.wind-wins {
            background: rgba(96, 165, 250, 0.15);
            color: #60a5fa;
            border: 1px solid #60a5fa;
        }

        @keyframes celebrate {
            0% { transform: scale(0.8); opacity: 0; }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); opacity: 1; }
        }

        /* Mobile Responsive Styles */
        @media (max-width: 768px) {
            body {
                padding: 15px;
            }

            .header {
                flex-direction: column;
                gap: 20px;
                text-align: center;
            }

            .header-stats {
                gap: 15px;
                flex-wrap: wrap;
                justify-content: center;
            }

            .stat-value {
                font-size: 1.2rem;
            }

            .main-grid {
                grid-template-columns: 1fr;
                gap: 15px;
            }

            .sidebar {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
            }

            .info-panel {
                margin-top: 0;
            }

            .board-grid {
                grid-template-columns: 24px repeat(5, 54px);
                grid-template-rows: 24px repeat(5, 54px);
                gap: 2px;
            }

            .cell {
                border-radius: 6px;
            }

            .dandelion-icon {
                font-size: 1.5rem;
            }

            .seed-icon {
                width: 10px;
                height: 10px;
            }

            .compass-display {
                width: 150px;
                height: 150px;
                margin-bottom: 15px;
            }

            .compass-center {
                width: 40px;
                height: 40px;
                font-size: 1.2rem;
            }

            .dir-btn {
                width: 32px;
                height: 32px;
                font-size: 0.6rem;
            }

            .dir-NE { top: 18px; right: 18px; }
            .dir-SE { bottom: 18px; right: 18px; }
            .dir-SW { bottom: 18px; left: 18px; }
            .dir-NW { top: 18px; left: 18px; }

            .turn-display {
                padding: 15px;
            }

            .turn-player {
                font-size: 1rem;
            }

            .action-buttons {
                flex-direction: column;
                gap: 8px;
                padding: 12px 15px;
            }

            .rules-panel .panel-content {
                padding: 15px;
            }

            .rules-panel ul {
                font-size: 0.85rem;
            }
        }

        @media (max-width: 480px) {
            .header-stats {
                gap: 10px;
            }

            .stat {
                min-width: 60px;
            }

            .stat-value {
                font-size: 1rem;
            }

            .stat-label {
                font-size: 0.65rem;
            }

            .sidebar {
                grid-template-columns: 1fr;
            }

            .board-grid {
                grid-template-columns: 20px repeat(5, 48px);
                grid-template-rows: 20px repeat(5, 48px);
            }

            .col-label, .row-label {
                font-size: 0.7rem;
            }

            .dandelion-icon {
                font-size: 1.3rem;
            }

            .seed-icon {
                width: 8px;
                height: 8px;
            }

            .compass-display {
                width: 130px;
                height: 130px;
            }

            .dir-btn {
                width: 28px;
                height: 28px;
                font-size: 0.55rem;
            }

            .dir-NE { top: 15px; right: 15px; }
            .dir-SE { bottom: 15px; right: 15px; }
            .dir-SW { bottom: 15px; left: 15px; }
            .dir-NW { top: 15px; left: 15px; }

            .panel-header {
                padding: 12px 15px;
            }

            .panel-title {
                font-size: 0.75rem;
            }

            .panel-badge {
                font-size: 0.65rem;
                padding: 3px 8px;
            }

            .legend-row {
                gap: 15px;
                padding: 12px;
            }

            .legend-item {
                font-size: 0.7rem;
            }

            .legend-icon {
                width: 20px;
                height: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <div class="logo">
                <div class="logo-icon">*</div>
                <h1>Dandelions</h1>
            </div>
            <div class="header-stats">
                <div class="stat">
                    <div class="stat-value" id="statTurn">1</div>
                    <div class="stat-label">Turn</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="statFlowers">0</div>
                    <div class="stat-label">Flowers</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="statSeeds">0</div>
                    <div class="stat-label">Seeds</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="statEmpty">25</div>
                    <div class="stat-label">Empty</div>
                </div>
            </div>
        </div>

        <div class="main-grid">
            <div class="panel board-panel">
                <div class="panel-header">
                    <span class="panel-title">Game Board</span>
                    <span class="panel-badge active" id="boardBadge">Dandelion's Turn</span>
                </div>
                <div class="panel-content">
                    <div class="board-container">
                        <div class="board-grid" id="boardGrid">
                            <!-- Board will be rendered by JavaScript -->
                        </div>
                    </div>
                </div>
                <div class="legend-row">
                    <div class="legend-item">
                        <div class="legend-icon dandelion">*</div>
                        <span>Dandelion</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon seed"><span></span></div>
                        <span>Seed</span>
                    </div>
                </div>
            </div>

            <div class="sidebar">
                <div class="panel compass-panel">
                    <div class="panel-header">
                        <span class="panel-title">Wind Control</span>
                        <span class="panel-badge waiting" id="compassBadge">Waiting</span>
                    </div>
                    <div class="panel-content">
                        <div class="compass-display">
                            <div class="compass-ring"></div>
                            <div class="compass-center">&#128168;</div>
                            {% for dir in directions_clockwise %}
                            <button class="dir-btn dir-{{ dir }}" data-dir="{{ dir }}">{{ dir }}</button>
                            {% endfor %}
                        </div>
                    </div>
                </div>

                <div class="panel info-panel">
                    <div class="turn-display">
                        <div class="current-turn">Current Turn</div>
                        <div class="turn-player" id="turnPlayer">Dandelion</div>
                    </div>
                    <div class="action-buttons">
                        <button class="btn btn-primary" id="newGameBtn">New Game</button>
                    </div>
                </div>
            </div>
        </div>

        <div id="winnerAnnouncement"></div>

        <div class="panel rules-panel">
            <div class="panel-header">
                <span class="panel-title">How to Play</span>
            </div>
            <div class="panel-content">
                <p class="rules-subtitle">A game of seeds and wind from "Math Games with Bad Drawings"</p>
                <h3>Rules</h3>
                <ul>
                    <li><strong>Dandelion</strong> places a flower (*) anywhere on the grid by clicking a cell.</li>
                    <li><strong>Wind</strong> blows in one of 8 directions by clicking a compass button.</li>
                    <li>When wind blows, seeds spread from ALL dandelions in that direction.</li>
                    <li>Each wind direction can only be used once!</li>
                    <li>After {{ num_turns }} turns each: <strong>Dandelions win</strong> if the board is full, otherwise <strong>Wind wins</strong>!</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        const BOARD_HEIGHT = {{ board_height }};
        const BOARD_WIDTH = {{ board_width }};
        const NUM_TURNS = {{ num_turns }};
        const DIRECTION_NAMES = {{ direction_names | tojson }};
        const COL_LABELS = {{ col_labels | tojson }};

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
            if (gameState.board[row][col] === 1) return;

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
            // Update stats
            const displayTurn = Math.min(gameState.turn + 1, NUM_TURNS);
            document.getElementById('statTurn').textContent = displayTurn;
            document.getElementById('statFlowers').textContent = gameState.flowers;
            document.getElementById('statSeeds').textContent = gameState.seeds;
            document.getElementById('statEmpty').textContent = gameState.empty;

            // Render entire board grid
            const boardGrid = document.getElementById('boardGrid');
            boardGrid.innerHTML = '';

            // Add corner cell
            const corner = document.createElement('div');
            corner.className = 'corner';
            boardGrid.appendChild(corner);

            // Add column labels
            for (let col = 0; col < BOARD_WIDTH; col++) {
                const colLabel = document.createElement('div');
                colLabel.className = 'col-label';
                colLabel.textContent = COL_LABELS[col];
                boardGrid.appendChild(colLabel);
            }

            // Add rows with row labels and cells
            for (let row = 0; row < BOARD_HEIGHT; row++) {
                // Add row label
                const rowLabel = document.createElement('div');
                rowLabel.className = 'row-label';
                rowLabel.textContent = row + 1;
                boardGrid.appendChild(rowLabel);

                // Add cells for this row
                for (let col = 0; col < BOARD_WIDTH; col++) {
                    const cell = document.createElement('div');
                    cell.className = 'cell';
                    const value = gameState.board[row][col];

                    if (value === 1) {
                        cell.classList.add('dandelion');
                        cell.innerHTML = '<span class="dandelion-icon">*</span>';
                    } else if (value === 2) {
                        cell.classList.add('seed');
                        cell.innerHTML = '<span class="seed-icon"></span>';
                    }

                    if (gameState.phase === 'dandelion' && !gameState.game_over && value !== 1) {
                        cell.classList.add('clickable');
                        cell.addEventListener('click', () => placeDandelion(row, col));
                    }

                    boardGrid.appendChild(cell);
                }
            }

            // Update compass buttons
            document.querySelectorAll('.dir-btn').forEach(btn => {
                const dir = btn.dataset.dir;
                const dirIndex = DIRECTION_NAMES.indexOf(dir);
                const isUsed = gameState.used_directions[dirIndex] === 1;
                btn.disabled = isUsed || gameState.phase !== 'wind' || gameState.game_over;
                btn.classList.toggle('used', isUsed);
            });

            // Update badges and turn display
            const boardBadge = document.getElementById('boardBadge');
            const compassBadge = document.getElementById('compassBadge');
            const turnPlayer = document.getElementById('turnPlayer');
            const winnerAnnouncement = document.getElementById('winnerAnnouncement');

            if (gameState.game_over) {
                if (gameState.winner === 'dandelion') {
                    boardBadge.textContent = 'Game Over';
                    boardBadge.className = 'panel-badge active';
                    compassBadge.textContent = 'Game Over';
                    compassBadge.className = 'panel-badge active';
                    turnPlayer.textContent = 'Dandelion Wins!';
                    turnPlayer.className = 'turn-player';
                    winnerAnnouncement.innerHTML = '<div class="winner-announcement dandelion-wins">The Dandelions Win! The meadow is covered!</div>';
                } else {
                    boardBadge.textContent = 'Game Over';
                    boardBadge.className = 'panel-badge waiting';
                    compassBadge.textContent = 'Game Over';
                    compassBadge.className = 'panel-badge waiting';
                    turnPlayer.textContent = 'Wind Wins!';
                    turnPlayer.className = 'turn-player wind';
                    winnerAnnouncement.innerHTML = '<div class="winner-announcement wind-wins">The Wind Wins! Empty spots remain!</div>';
                }
            } else {
                winnerAnnouncement.innerHTML = '';
                if (gameState.phase === 'dandelion') {
                    boardBadge.textContent = "Dandelion's Turn";
                    boardBadge.className = 'panel-badge active';
                    compassBadge.textContent = 'Waiting';
                    compassBadge.className = 'panel-badge waiting';
                    turnPlayer.textContent = 'Dandelion';
                    turnPlayer.className = 'turn-player';
                } else {
                    boardBadge.textContent = 'Waiting';
                    boardBadge.className = 'panel-badge waiting';
                    compassBadge.textContent = "Wind's Turn";
                    compassBadge.className = 'panel-badge active';
                    turnPlayer.textContent = 'Wind';
                    turnPlayer.className = 'turn-player wind';
                }
            }
        }

        // Event listeners
        document.getElementById('newGameBtn').addEventListener('click', newGame);
        document.querySelectorAll('.dir-btn').forEach(btn => {
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

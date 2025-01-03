EPOCHS = 3250000
GAMMA = 0.98
reward_vals = {'win': 100, 'illegal': -100, 'lose': -85, 'meh': 3}
LEARNING_RATE = 0.002
EXPLORATION_PROB_STEPS = {10.0: 0.2, 12.0: 0.1, 15.0: 0.0, 17.0: 0.1, 18.0: 0.0, 18.5: 0.05, 28.0: 0.0, 33.5: 0.08, 42.0: 0.0, 44.5: 0.05, 48.0: 0.0}
INPUT_SIZE = 58
HIDDEN_SIZE = 116
OUTPUT_SIZE = 25
MIDDLE_LAYERS = [116, 116, 116]
device = 'cpu'

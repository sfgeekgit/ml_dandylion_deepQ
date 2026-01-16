"""
Utility functions for model discovery and parameter loading.
Used by the web interface to dynamically find available trained models.
"""
import os


def discover_models(model_type):
    """
    Scan models directory for available models.

    Args:
        model_type: 'seeds' or 'wind'

    Returns:
        List of dicts: [{'id': '007', 'path': 'models/seeds/007', 'model_file': 'seedbrain.pth'}, ...]
    """
    base_dir = f"models/{model_type}"
    model_filename = "seedbrain.pth" if model_type == "seeds" else "windbrain.pth"
    models = []

    if not os.path.exists(base_dir):
        return []

    for subdir in sorted(os.listdir(base_dir)):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        model_path = os.path.join(subdir_path, model_filename)
        if os.path.exists(model_path):
            models.append({
                'id': subdir,
                'path': subdir_path,
                'model_file': model_filename
            })

    return models


def load_params(model_type, model_id):
    """
    Load hyperparameters from paramsXXX.py file.

    Args:
        model_type: 'seeds' or 'wind'
        model_id: directory name (e.g., '007')

    Returns:
        Dict of parameters or None if not found
    """
    base_dir = f"models/{model_type}/{model_id}"
    params_file = f"params{model_id}.py"
    params_path = os.path.join(base_dir, params_file)

    if not os.path.exists(params_path):
        return None

    params = {}
    with open(params_path, 'r') as f:
        content = f.read()
        # Parse Python assignments safely
        for line in content.strip().split('\n'):
            if '=' in line and not line.strip().startswith('#'):
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()
                try:
                    # eval is safe here since we control the params files
                    params[key] = eval(value)
                except:
                    params[key] = value

    return params


def format_params_for_display(params):
    """
    Format parameters for UI display.
    Returns all params in a display-friendly format.
    """
    if not params:
        return {}

    # Format epochs with commas
    epochs = params.get('EPOCHS', 'N/A')
    if isinstance(epochs, int):
        epochs_display = f"{epochs:,}"
    else:
        epochs_display = str(epochs)

    # Calculate layer count and architecture string
    middle_layers = params.get('MIDDLE_LAYERS', [])
    input_size = params.get('INPUT_SIZE', '?')
    output_size = params.get('OUTPUT_SIZE', '?')

    layer_count = len(middle_layers) + 2  # input + hidden layers + output

    arch_parts = [str(input_size)] + [str(x) for x in middle_layers] + [str(output_size)]
    architecture = ' -> '.join(arch_parts)

    display = {
        'epochs': epochs_display,
        'layer_count': layer_count,
        'architecture': architecture,
        'learning_rate': params.get('LEARNING_RATE', 'N/A'),
        'gamma': params.get('GAMMA', 'N/A'),
        'rewards': params.get('reward_vals', {}),
        'exploration_steps': params.get('EXPLORATION_PROB_STEPS', {}),
        'input_size': input_size,
        'output_size': output_size,
        'middle_layers': middle_layers,
        'hidden_size': params.get('HIDDEN_SIZE', 'N/A'),
        'device': params.get('device', 'N/A'),
        'raw': params  # Keep raw params for full display
    }
    return display


def models_available():
    """Check if any AI models exist for either role."""
    return bool(discover_models('seeds')) or bool(discover_models('wind'))


def seed_models_available():
    """Check if any seed (dandelion) models exist."""
    return bool(discover_models('seeds'))


def wind_models_available():
    """Check if any wind models exist."""
    return bool(discover_models('wind'))

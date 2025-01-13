import streamlit as st
import os
from game import *
from brainlib import * # board_state_to_tensor #, board_state_from_tensor
from cvc import model_v_model


st.markdown(f"""Neural Networks trained with Deep Q Learning to play the game of Dandelion.\n\n
Dandelion is an asymmetric two player game featured in the book \"Math Games With Bad Drawings\" by Ben Orlin.
        One player plays the role of  Dandelion Seeds and the other is the Wind.""")

st.markdown(f"""The Dandelion places seeds on the board and the Wind blows the seeds around.
        If the board is filled with seeds, the Dandelion wins.
        If there are still empty spots after seven turns, the Wind wins.""")


seed_model_files, wind_model_files = [], []
seeds_dir, wind_dir = "models/seeds", "models/wind"

# Get list of model files from models/seeds directory
for subdir in os.listdir(seeds_dir):
    model_path = os.path.join(seeds_dir, subdir, "seedbrain.pth")
    if os.path.exists(model_path):
        seed_model_files.append(f"{subdir}/seedbrain.pth")

# Get list of model files from models/wind directory
for subdir in os.listdir(wind_dir):
    model_path = os.path.join(wind_dir, subdir, "windbrain.pth")
    if os.path.exists(model_path):
        wind_model_files.append(f"{subdir}/windbrain.pth")

def display_model_params(selected_model, model_type, base_dir):
    """Display parameters for a selected model
    model_type: 'seed' or 'wind' for display text
    base_dir: 'models/seeds' or 'models/wind'"""
    
    if not selected_model:
        return
        
    role = "dandelion" if model_type == "seed" else "wind"
    model_num = selected_model.split('/')[0]
    params_path = os.path.join(base_dir, model_num, f"params{model_num}.py")
    
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params_content = f.read()
            # Find EPOCHS value and layer count
            for line in params_content.split('\n'):
                if line.startswith('EPOCHS'):
                    epochs = line.split('=')[1].strip()
                if line.startswith('MIDDLE_LAYERS'):
                    layers = eval(line.split('=')[1].strip())

            # Filter out device line and create cleaned content
            cleaned_params = '\n'.join(line for line in params_content.split('\n') 
                                     if not line.strip().startswith('device'))

            st.markdown(f"""<p style='margin-bottom:0px'><b>{role}</b><BR>You selected: {selected_model}</p>""", unsafe_allow_html=True)  
            st.markdown(f"""It was trained for {int(epochs):,} epochs and has {len(layers)+2} layers""")
            st.code(cleaned_params, language='python')
    else:
        st.warning(f"No params{model_num}.py found for {selected_model}")

# Create dropdowns
selected_seed = st.selectbox(
    "Select a seed model",
    seed_model_files,
    index=0 if seed_model_files else None,
    placeholder="Choose a model..."
)

selected_wind = st.selectbox(
    "Select a wind model",
    wind_model_files,
    index=0 if wind_model_files else None,
    placeholder="Choose a model..."
)

# Create two columns for side-by-side display
col1, col2 = st.columns(2)

# Display parameters for both models in separate columns
with col1:
    display_model_params(selected_seed, "seed", "models/seeds")
    
with col2:
    display_model_params(selected_wind, "wind", "models/wind")



#st.markdown(f"""Now, get this working on my local machine as a next step""")

# for local running
seedbrain_dir      = "./models/seeds/" 
windbrain_dir =      "./models/wind/"

seedbrain = load_model(seedbrain_dir, selected_seed)
windbrain = load_model(windbrain_dir, selected_wind)

# input to allow user to set temperature
wind_temp = st.slider("Wind Temperature", 0.0, 20.0, 2.0)
seed_temp = st.slider("Seed Temperature", 0.0, 20.0, 2.0)

# after loading models, display a button to run the game
if st.button("Run Game"):
    winner, out_str = model_v_model(seedbrain, windbrain, seed_temp, wind_temp)
    st.write(out_str)

# Run model vs model game
#winner, out_str = model_v_model(seedbrain, windbrain)
#st.write(out_str)


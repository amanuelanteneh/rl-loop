import numpy as np
import os
import sys
import yaml
from circuit import Circuit
from stable_baselines3 import PPO
from typing import Dict, Any, List
from utils import episode_stats, get_states, histogram

num_eval_episodes = int(sys.argv[1])
model_name = sys.argv[2]
verify = sys.argv[2] == 't'

print("Model name: ", model_name, flush=True)

with open("models/"+ model_name +"/training-parameters.yml", 'r') as file:
         training_parameters = yaml.safe_load(file)
    
circuit_parameters: Dict[str, Any] = training_parameters['circuit']
model_parameters: Dict[str, Any] = training_parameters['model']
deterministic = True

os.makedirs('evals/'+model_name, exist_ok=True)

final_fidelities = []
photon_counts = []
steps_no_resets = []
steps_resets = []

intervals = np.linspace(1, num_eval_episodes, 6, dtype=int) # to change random seed of env at 5 intervals of evalution
lstm_states = None # used if agent has LSTM layer
episode_starts = np.ones((1,), dtype=bool) # used if agent has LSTM layer
seed = 0

target = model_name.split('_')[11]
state = target.split('~')[0]
state_params = target.split('~')[1:]
target_states: List[np.ndarray] = get_states(state, circuit_parameters["hilbert_dimension"], state_params)

plot_env = Circuit(circuit_parameters, targets=target_states, seed=42, evaluate=False)
# plot the target state
plot_env.render(name=None, filename="evals/"+model_name+"/target", is_target=True)
del plot_env # no longer needed

for i in range(1, num_eval_episodes+1):
    if i in intervals:
        seed += 1
        # create evaluation environment
        env = Circuit(circuit_parameters, targets=target_states, seed=seed, evaluate=True)
        
        if verify:
            verify_params = circuit_parameters
            verify_params["hilbert_dimension"] = 40
            env_verify = Circuit(verify_params, targets=target_states, seed=seed, evaluate=True)

        model = PPO.load('models/'+model_name+"/rl_model", env=env)
        print(f'Creating new evaluation environment with random seed: {seed}')
    
    state = env.reset()
    if verify:
       state_verify = env_verify.reset()

    done = False
    while not done:
        if model_parameters["use_lstm"]:
            action, lstm_states = model.predict(state, state=lstm_states, episode_start=episode_starts, deterministic=True)
            action, _states = model.predict(state, deterministic=deterministic)
            state, r, done, info = env.step(action)
            episode_starts = done
            if verify:
                n = info['n']
                state_verify, _, _, info = env_verify.step(action, postselect=n)
        else:
          action, _ = model.predict(observation=state, deterministic=deterministic)
          state, r, done, info = env.step(action)
          if verify:
            n = info['n']
            state_verify, _, _, _ = env_verify.step(action, postselect=n)

    final_fidelities.append(env.steps[-1]['F'])
    sr, snr, pn = episode_stats(env.steps)
    steps_no_resets.append(snr)
    steps_resets.append(sr)
    photon_counts.append(pn)

    env.render(name="episode "+str(i)+" state", 
               filename="evals/"+model_name+"/plot-"+str(i), is_target=False)
    if verify:
        env.render(name="episode "+str(i)+" state", 
                   filename="evals/"+model_name+"/verify-plot-"+str(i), is_target=False)

final_fidelities = np.array(final_fidelities)

print("Max steps:", circuit_parameters["max_steps"], "\n")
print("Total number of episodes:", num_eval_episodes, "\n")
fidelity_thresholds = [.89, .9, .925, .95, .96, .97, .98, .99]
for f in fidelity_thresholds:
    print(f"Number of episodes with final fidelity {f*100}% of more:", (final_fidelities >= f).sum())

num_bins = 200

histogram(num_bins, final_fidelities, steps_resets, steps_no_resets,
          photon_counts, num_eval_episodes, model_name, 
          circuit_parameters["max_steps"], filename="stats-histogram")
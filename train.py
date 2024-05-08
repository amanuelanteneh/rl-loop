from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import RecurrentPPO
import numpy as np
import torch
import torch.optim as optim
import sys
from circuit import Circuit
from utils import  EpisodeCallbackMulti,\
                   TimestepCallbackMulti, CheckpointCallback, get_states
import os
import yaml
import shutil
from typing import Callable, Dict, List, Any


def make_env(env_parameters: Dict[str, str], targets: np.ndarray ,rank: int, seed: int) -> Callable:
    """
    Utility function for multiprocessed env
    
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> Circuit:
        env = Circuit(env_parameters, targets=targets, seed=seed+rank, evaluate=False)
       
        return env

    return _init


if __name__ == '__main__': # needed for multi proc
    
    with open('training-parameters.yml', 'r') as file:
         training_parameters = yaml.safe_load(file)
    
    model_parameters: Dict[str, Any] = training_parameters['model']
    circuit_parameters: Dict[str, Any] = training_parameters['circuit']

    print("\nModel parameters:\n", flush=True)
    for key in model_parameters:
        print(key, ":", model_parameters[key], flush=True)
    
    print("\nCircuit parameters:\n", flush=True)
    for key in circuit_parameters:
        print(key, ":", circuit_parameters[key], flush=True)

    
    cpus = int(sys.argv[1]) # number of environments to run in parallel
    target = sys.argv[2]

    # model parameters
    activation: str = model_parameters["activation_func"]
    use_lstm: bool = model_parameters["use_lstm"]
    gamma: float = model_parameters["gamma"]
    clip_range: float = model_parameters["clip_range"]
    learning_rate: float = model_parameters["learning_rate"]
    num_epochs: int = model_parameters["num_epochs"]
    batch_size: int = model_parameters["batch_size"]
    buffer_size: int = model_parameters["buffer_size"]
    total_timesteps: int = model_parameters["total_timesteps"]
    hidden_layers: List[int] = model_parameters["hidden_layers"]
    

    model_name = 'dim_'+str(circuit_parameters["hilbert_dimension"]) \
                +'_exp_'+str(circuit_parameters["penalty_exponent"])\
                +'_rew_'+circuit_parameters["reward_func"]\
                +'_maxsqz_'+str(circuit_parameters["max_sqz"])\
                +'_dmax_'+str(circuit_parameters["max_disp"])\
                +'_tar_'+target+'_type_'+circuit_parameters["circuit_type"]\
                +'_loss_'+str(circuit_parameters["loss"])

    os.makedirs('models/', exist_ok=True) # create folder for models if not already there

    os.makedirs('models/' + model_name, exist_ok=True) # create folder for agent with these parameters
    
    shutil.copyfile('training-parameters.yml', 'models/' + model_name + '/training-parameters.yml')

    log_dir = "logs/"

    os.makedirs(log_dir, exist_ok=True)
    
    # get target states
    state = target.split('~')[0]
    state_params = target.split('~')[1:]
    target_states: List[np.ndarray] = get_states(state, circuit_parameters["hilbert_dimension"], state_params)

    # create env to plot initial state 
    plov_env = Circuit(circuit_parameters, targets=target_states, seed=42, evaluate=False)

    # plot the initial state
    plov_env.render(name='initial', filename='models/'+model_name+"/start", is_target=False)
    # plot the target state
    plov_env.render(name=None, filename='models/'+model_name+"/target", is_target=True)
    
    del plov_env # no longer needed

    multi_proc = os.cpu_count() >= cpus
    if multi_proc:
    
        n_steps = buffer_size // cpus
        print(f"\nUsing multi-proccessing with {cpus} cpu cores ({cpus} environments)\n")
        print(f"n_steps passed to PPO object was changed to be different than that given in training-parameters yaml file, now is: {n_steps}\n")
        
        # create env vector for parallel training
        env = SubprocVecEnv([make_env(circuit_parameters, targets=target_states, rank=i, seed=42+i) for i in range(cpus)])
        
        checkpoint_callback = CheckpointCallback(save_freq=max(50_000 // cpus, 1), 
                                                            save_path="models/"+model_name, 
                                                            name_prefix="rl_model")
        
        timestep_callback = TimestepCallbackMulti()
        eps_callback = EpisodeCallbackMulti()
    
    else:
        print("\nNumber of requested environments does not match or exceed the number of cores on machine.")
        exit()

    # create RL model
    if activation == 'relu':
       act = torch.nn.ReLU
    else:
       act = torch.nn.Tanh

    if not use_lstm: # not using LSTM layer
        # pi is neural network arch of the actor and vf is arch for the critic
        policy_kwargs = dict(activation_fn = act,
                         net_arch=dict(pi=hidden_layers, 
                                       vf=hidden_layers), 
                         optimizer_class = optim.Adam)

        model = PPO("MlpPolicy",
                    env,
                    gamma=gamma,
                    n_epochs=num_epochs,
                    batch_size=batch_size,
                    clip_range=clip_range,
                    n_steps=n_steps,
                    learning_rate=learning_rate,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=log_dir)

    else: # if we want the first layer of the actor and critic networks to be LSTM layers
         policy_kwargs = dict(activation_fn = act,
                         net_arch=dict(pi=hidden_layers, 
                                       vf=hidden_layers), 
                         lstm_hidden_size = hidden_layers[0], 
                         n_lstm_layers = 1,
                         optimizer_class = optim.Adam)

         model = RecurrentPPO("MlpLstmPolicy",
                    env,
                    gamma=gamma,
                    n_epochs=num_epochs,
                    batch_size=batch_size,
                    clip_range=clip_range,
                    n_steps=n_steps,
                    learning_rate=learning_rate,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=log_dir) 

    print("\nNeural network architecture: \n\n", model.policy, flush=True)

    print("\nStarting training.", flush=True)

    model.learn(total_timesteps=total_timesteps, tb_log_name=model_name, callback=[timestep_callback, eps_callback, checkpoint_callback])

    print("\nTraining complete.")
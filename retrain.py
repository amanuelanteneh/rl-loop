from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from sb3_contrib import RecurrentPPO
import numpy as np
import sys
from circuit import Circuit
from utils import  EpisodeCallbackMulti,\
                   TimestepCallbackMulti, CheckpointCallback, get_states
import os
import yaml
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
    cpus = int(sys.argv[1])
    model_name = sys.argv[2]

    model_path = "models/" + model_name + "/rl_model.zip"
    model_logs = "logs/" + model_name

    with open("models/" + model_name + '/parameters.yml', 'r') as file:
         training_parameters = yaml.safe_load(file)
    
    model_parameters: Dict[str, Any] = training_parameters['model']
    circuit_parameters: Dict[str, Any] = training_parameters['circuit']
    print("\nCircuit parameters:\n", flush=True)
    for key in circuit_parameters:
        print(key, ":", circuit_parameters[key], flush=True)
    
    # get target states, next 3 lines are just done so the target state can be grabbed from the model_name
    model_list: List[str] = model_name.split("_")
    model_dict: Dict[str, str] = { model_list[i] : model_list[i+1] for i in range(0, len(model_list), 2) }
    target = model_dict["tar"]
    state = target.split('~')[0]
    state_params = target.split('~')[1:]
    target_states: List[np.ndarray] = get_states(state, circuit_parameters["hilbert_dimension"], state_params)

    # some model/learning parameters
    total_timesteps: int = model_parameters["total_timesteps"]

    multi_proc = os.cpu_count() >= cpus
    if multi_proc:
        print(f"\nUsing multi-processing with {cpus} cpu cores ({cpus} environments)\n")
        
        # create env vector for parallel training
        env = SubprocVecEnv([make_env(circuit_parameters, targets=target_states, rank=i, seed=42+i) for i in range(cpus)])
        save_freq = max(model_parameters["save_freq"] // cpus, 1)
        checkpoint_callback = CheckpointCallback(save_freq=save_freq, 
                                                            save_path="models/"+model_name, 
                                                            name_prefix="rl_model")
        
        timestep_callback = TimestepCallbackMulti()
        eps_callback = EpisodeCallbackMulti()
    else:
        print("\nNumber of requested environments does not match or exceed the number of cores on machine.")
        exit()

    if model_parameters["use_lstm"] == True:
        model = RecurrentPPO.load(model_path, print_system_info=True)
    else:
        model = PPO.load(model_path, print_system_info=True)

    model.set_env(env)
    log_dir = "logs/"
    # appending _1 to folder name because that's how sb3 creates the initial logger folder
    # the logger will now create a new events.out.tfevents file in the log folder for the model
    # and tensorboard will just concatenate them when displaying the training curves

    new_logger = configure(log_dir + model_name + "_1", ["tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps, callback=[timestep_callback, eps_callback, checkpoint_callback], reset_num_timesteps=False)

    exit()

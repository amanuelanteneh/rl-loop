from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import RecurrentPPO
import torch
import torch.optim as optim
import sys
from circuit import Circuit
from utils import *
import os


if __name__ == '__main__': # needed for multi proc
    
    parameters = {}
    with open("training-parameters.txt") as file: # read training file
        for line in file:
           (key, val) = line.split()
           parameters[key] = val
 
    print("Parameters used for training:", flush=True)
    for key in parameters:
        print(key, ":", parameters[key], flush=True)
        

    if activation == 'relu':
       act = torch.nn.ReLU
    else:
       act = torch.nn.Tanh


    model_name = 'dim_'+str(dim) \
                +'_sqz0_'+parameters['initial-sqz']\
                +'_exp_'+str(exp)\
                +'_rew_'+reward+'_sqzmax_'+parameters['sqz-max']\
                +'_dmax_'+parameters['disp-max']
                +'_tar_'+str(target)+'_lstm_'+parameters['lstm']\
                +'_pnr_'+parameters['pnr-disp-mag']\
                +'_loss_'+parameters['loss']\
                +'_buf_'+str(n_steps)+'_epoch_'+str(n_epochs)\
                +'_batch_'+str(batchSize)+'_lr_'+str(lr)

    os.makedirs('models/', exist_ok=True) # create folder for models if not already there

    os.makedirs('models/' + model_name, exist_ok=True) # create folder for agent with these parameters

    log_dir = "logs/"

    os.makedirs(log_dir, exist_ok=True)
    
    if multiProc:
    
        cpus = int(sys.argv[1])
        n_steps = n_steps // cpus
        print(f"\nUsing multi-proccessing with {cpus} cpu cores ({cpus} environments)\n")
        print(f"n_steps passed to PPO object was changed to be different than that given in model-parameters file, now is: {n_steps}\n")
        
        # create env vector for parallel training
        env = SubprocVecEnv([make_env(dim, intialSqz, maxSteps, exp, sqzMax, dispMax, pnr_disp, tune_pnr_phi, target,\
                                  reward, i, 2023, loss) for i in range(cpus)])


        # create env to plot inital state and evaluate agent
        plotEnv = Circuit(dim, intialSqz, maxSteps, exp, sqzMax, dispMax, pnr_disp, tune_pnr_phi, target, \
                                   reward, 1982, False, loss)

        # plot the initial state
        plotEnv.render(False, 'initial', 'models/'+modelName+"/start")
        # plot the target state
        plotEnv.render(True, 'target', 'models/'+modelName+"/target")
        
        del plotEnv # no longer needed
        
        checkpoint_callback = CheckpointCallback(save_freq=max(50_000 // cpus, 1), save_path="models/"+modelName, name_prefix="rl_model")
        
        timestep_callback = TimestepCallbackMulti()
        eps_callback = EpisodeCallbackMulti()
        
    else:
        print("\nNot using multi-processing")
        timestep_callback = TimestepCallback()
        eps_callback = EpisodeCallback()
        
        checkpoint_callback = CheckpointCallback(save_freq=2000, 
                                                 save_path="models/"+modelName, name_prefix="rl_model")
        
        env = Circuit(dim, intialSqz, maxSteps, exp, sqzMax, dispMax, pnr_disp, tune_pnr_phi, target,\
                                   reward, 2023, False, loss)
        
         # plot the initial state
        env.render(False, 'initial', 'models/'+modelName+"/start")
        # plot the target state
        env.render(True, 'target', 'models/'+modelName+"/target")


    if not useLSTM: # not using LSTM layer
        # pi is neural network arch of the actor and vf is arch for the critic
        policy_kwargs = dict(activation_fn = act,
                         net_arch=dict(pi=[hidden1, hidden2, hidden3], vf=[hidden1, hidden2, hidden3]), 
                         optimizer_class = optim.Adam)

        model = PPO("MlpPolicy",
                    env,
                    gamma=gamma,
                    n_epochs=n_epochs,
                    batch_size=batchSize,
                    clip_range=clip_range,
                    n_steps=n_steps,
                    learning_rate=lr,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=logDir)

    else: # if we want the first layer of the actor and critic networks to be LSTM layers
         policy_kwargs = dict(activation_fn = act,
                         net_arch=dict(pi=[hidden1, hidden2, hidden3], vf=[hidden1, hidden2, hidden3]), 
                         lstm_hidden_size = hidden1, 
                         n_lstm_layers = 1,
                         optimizer_class = optim.Adam)

         model = RecurrentPPO("MlpLstmPolicy",
                    env,
                    gamma=gamma,
                    n_epochs=n_epochs,
                    batch_size=batchSize,
                    clip_range=clip_range,
                    n_steps=n_steps,
                    learning_rate=lr,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=logd_ir) 

    print("\nNeural network architecture: \n\n", model.policy, flush=True)

    print("\nStarting training.", flush=True)

    if not multiProc:
        model.learn(total_timesteps=totalTimesteps, tb_log_name=modelName, callback=[timestep_callback, eps_callback, checkpoint_callback])
    else:
        # model.learn(total_timesteps=totalTimesteps, tb_log_name=modelName, callback=[timestep_callback, eps_callback, checkpoint_callback])
        model.learn(total_timesteps=totalTimesteps, tb_log_name=modelName, callback=[timestep_callback, eps_callback, checkpoint_callback])

    print("\nTraining complete.")
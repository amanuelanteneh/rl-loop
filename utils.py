import numpy as np
from numpy import pi, sqrt, cosh, exp, tanh, outer, arange, meshgrid, real, diag

from qutip import wigner, squeeze, displace, momentum, position, fock, Qobj, coherent, destroy

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

import matplotlib.pyplot as plt
from matplotlib import cm

import math

from typing import List, Union, Tuple, Dict
import os


def squeezed_vacuum(r, theta, dim, dtype=np.complex128) -> np.ndarray:
    const = 1.0 / sqrt(cosh(r))
    state = const * np.array( [ ((-exp(1j*theta)*tanh(r))**(n//2)) *  sqrt(math.gamma(2*(n//2) + 1)) / ((2**(n//2)) * math.gamma(n//2 + 1)) 
            		            if n % 2 == 0 else 0.0 + 0.0j for n in range(dim) ], dtype=dtype )
    norm = np.linalg.norm(state)
    state /= norm # normalize
    dm = outer(state, state.conj().T)

    return dm

def plot_state(dm: np.ndarray, max_quad: float, points: int, dim: int, 
               name: str, fid: float = None, filename:str = 'default') -> None: 
        
        font = 10.5
        fig = plt.figure(figsize=(13.5/1.25, 6/1.25))
        plt.rcParams.update({'font.size': font})

        dx = max_quad / points
        quad_vals = arange(-points, points+1) * dx
        Q, P = meshgrid(quad_vals, quad_vals)  
        W = wigner(Qobj(dm),quad_vals, quad_vals)
        
        ax = fig.add_subplot(1, 2, 1)
        wigner_func = ax.contourf(Q, P, W, levels=70, cmap=cm.RdBu, vmin=-1/pi, vmax=1/pi)
        ax.set_ylabel('P', fontsize=font)
        ax.set_xlabel('Q', fontsize=font)
        cb = fig.colorbar(wigner_func, shrink = 0.8) # add colorbar
        
        if fid is None:
           ax.set_title("Wigner function of \n"+name+".")
        else:
           ax.set_title("Wigner function of \n"+name+".\n Fidelity: "+str(round(fid*100, 2)) + "%" )
        
        

        ax = fig.add_subplot(1, 2, 2)
        photon_probs = real(diag(dm))
        avg_n = sum([ i*photon_probs[i] for i in range(dim) ])
        trace = sum(photon_probs)
        photon_dist = ax.bar(range(dim), photon_probs)
        
        ax.set_xticks(range(0, dim+1, 2))
        
        ax.set_ylabel('P(n)', fontsize=font)
        ax.set_xlabel('n', fontsize=font)
        ax.set_title("Photon number distribution of\n" f"{name}." + "\n" + r"$\langle n \rangle$" +\
                     f": {round(avg_n, 3)}" + r"     Tr$[\rho]$" + f": {round(trace, 4)}" )
        
        fig.tight_layout()
        plt.savefig(filename+".png", dpi=120)
        plt.clf()
        plt.close()
        return

def get_states(state_type: str, dim: int, state_params: List[Union[int, float]]) -> List[np.ndarray]:
    real_dim = dim # the actual dimension of the dm returned
    dim = 100 # higher cutoff to get better numerical result
    states = []

    if state_type == "cubic":
        x = position(dim)
        p = momentum(dim)
        cubicity = float(state_params[0])
        r = float(state_params[1])
        a = float(state_params[2])

        params = ( [p, r, a],
                   [-p, r, -a],
                   [x, -r, -1j*a],
                   [-x, -r, 1j*a] )
        
        for param in params:
            V = (1j*cubicity*(param[0]**3)).expm() # cubic phase gate
            S = squeeze(dim, param[1]) # squeeze gate
            D = displace(dim, param[2])
            psi = D*V*S*fock(dim, 0)
            ket = np.array(psi).flatten()[:real_dim] / np.linalg.norm( np.array(psi).flatten()[:real_dim] )
            dm = outer(ket, ket.conj().T)
            states.append(dm)

    elif state_type == "sqzcat":
        a = float(state_params[0])
        r = float(state_params[1])

        params = ( [a, r], [1j*a, -r] )
        
        for param in params:
            psi = squeeze(dim, param[1]) * ( coherent(dim, param[0]) + coherent(dim, -param[0]) ).unit()
            ket = np.array(psi).flatten()[:real_dim] / np.linalg.norm( np.array(psi).flatten()[:real_dim] )
            dm = outer(ket, ket.conj().T)
            states.append(dm)

            psi = squeeze(dim, param[1]) * ( coherent(dim, param[0]) - coherent(dim, -param[0]) ).unit()
            ket = np.array(psi).flatten()[:real_dim] / np.linalg.norm( np.array(psi).flatten()[:real_dim] )
            dm = outer(ket, ket.conj().T)
            states.append(dm)

    elif state_type == "super":
        first = int(state_params[0])
        second = int(state_params[1])
        psi = ( fock(dim, first) + fock(dim, second) ).unit()
        ket = np.array(psi).flatten()[:real_dim] / np.linalg.norm( np.array(psi).flatten()[:real_dim] )
        dm = outer(ket, ket.conj().T)
        states.append(dm)
        
        a = destroy(dim)
        R = (1j*(pi/2)*a.dag()*a).expm()
        psi = R * psi
        ket = np.array(psi).flatten()[:real_dim] / np.linalg.norm( np.array(psi).flatten()[:real_dim] )
        dm = outer(ket, ket.conj().T)
        states.append(dm)




    elif state_type == "cat":
        a = float(state_params[0])

        params = ( a, 1j*a )
        
        for param in params:
            psi = ( coherent(dim, param) + coherent(dim, -param) ).unit()
            ket = np.array(psi).flatten()[:real_dim] / np.linalg.norm( np.array(psi).flatten()[:real_dim] )
            dm = outer(ket, ket.conj().T)
            states.append(dm)

            psi = ( coherent(dim, param) - coherent(dim, -param) ).unit()
            ket = np.array(psi).flatten()[:real_dim] / np.linalg.norm( np.array(psi).flatten()[:real_dim] )
            dm = outer(ket, ket.conj().T)
            states.append(dm)

    return states

def episode_stats(ep):
    pnrs = 0
    steps_without_resets = 0
    steps_with_resets = 0
    for i in range(len(ep)):
        pnrs += ep[i]['n']
        steps_with_resets += 1
        steps_without_resets += 1
        if round(ep[i]['t'], 2) == 1.0:
           steps_without_resets = 0
        if round(ep[i]['t'], 2) == 0.0 and round(ep[i]['d-inline'], 2) == 0.0:
           break

    return steps_with_resets, steps_without_resets, pnrs

def steps_table(steps: List[ Dict[str, Union[float, int]] ], 
              max_steps: int, model_name: str, filename: str) -> None:
    
    plt.rcParams.update({'font.size': 17})
    fig = plt.figure(figsize=(15, 15), dpi=200)
    plt.axis('off')

    cell_data = []
    columns = (r'$F$', r'$t$', r'$r$', r'$d_{inline}$', r'$\phi_{inline}$',
               r'$d_{pnr}$', r'$\phi_{pnr}$',
               r'Tr$[\rho]$', r'$n$', r'$P(\vec{n})$' )
    rows = [ str(x) for x in range(1, max_steps+1) ]
    
    for i in range(len(steps)):
        cell_data.append(  [ round(steps[i]['F'], 3), round(steps[i]['t'], 3), round(steps[i]['r'], 3), 
                            round(steps[i]['d-inline'], 3), round(steps[i]['d-inline-phi'], 3),
                            round(steps[i]['d-pnr'], 3), round(steps[i]['d-pnr-phi'], 3),  
                            round(steps[i]['Tr'], 3), steps[i]['n'],  round(steps[i]['P'], 3)
                            ]  )

    table = plt.table(cellText=cell_data,
                      rowLabels=rows,
                      colLabels=columns,
                      colWidths=[1.0/(4*len(columns))]*len(columns),
                      loc='best')
    table.scale(4.0, 1.5)
    fig.tight_layout()
    plt.savefig("evals/"+model_name+"/"+filename+".png", dpi=200)
    plt.clf()
    plt.close()

    return

def histogram(num_bins, final_fidelities, steps_resets, steps_no_resets, photon_counts, 
              num_eval_episodes, model_name, max_steps, filename="stats-histogram") -> None:
    plt.rcParams.update({'font.size': 24})
    fig = plt.figure(figsize=(16, 8), dpi=180)
    
    ax = fig.add_subplot(2, 2, 1)
    bins = np.linspace(0, 1, num_bins)
    ax.axis(xmin=-0.1, xmax=1.1)
    ax.hist(final_fidelities, bins=bins, alpha=0.8)
    ax.set_xlabel(f'Output state fidelity' + '\n' + f"for {num_eval_episodes} episodes")
    
    ax = fig.add_subplot(2, 2, 2)
    bins = np.linspace(0, 350, 350+1, dtype=int)
    ax.axis(xmin=-2, xmax=350)
    ax.hist(photon_counts, bins=bins, alpha=0.8)
    ax.set_xlabel('Total photons detected per episode \n (50 detections per episode)')
    
    ax = fig.add_subplot(2, 2, 3)
    bins = np.linspace(0, max_steps, max_steps+1, dtype=int)
    ax.axis(xmin=-2, xmax=max_steps+1)
    ax.hist(steps_resets, bins=bins, alpha=0.8)
    ax.set_xlabel(f'Total steps per episode with resets')
    
    ax = fig.add_subplot(2, 2, 4)
    bins = np.linspace(0, max_steps, max_steps+1, dtype=int)
    ax.axis(xmin=-2, xmax=max_steps+1)
    ax.hist(steps_no_resets, bins=bins, alpha=0.8)
    ax.set_xlabel(f'Total steps per episode without resets')
    
    fig.tight_layout()
    plt.savefig('evals/'+model_name+"/eval-"+filename+".png", dpi=180)
    plt.clf()
    plt.close()

    return

class EpisodeCallbackMulti(BaseCallback):
    """
    Callback used for logging episode data.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Custom counter to reports stats
        # (and avoid reporting multiple values for the same step)
        self._episode_counter = 0
        self._tensorboard_writer = None

    def _init_callback(self) -> None:
        assert self.logger is not None
        # Retrieve tensorboard writer to not flood the logger output
        for out_format in self.logger.output_formats:
            if isinstance(out_format, TensorBoardOutputFormat):
                self._tensorboard_writer = out_format
        assert self._tensorboard_writer is not None, "You must activate tensorboard logging when using RawStatisticsCallback"

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for i in range(len(infos)):
            if self.locals['dones'][i]:
                self.logger.record(f"episode/final_fidelity", infos[i]["F"])
                self._episode_counter += 1

        self.logger.dump(self._episode_counter)                

        return True

class TimestepCallbackMulti(BaseCallback):
    """
    Callback used for logging timestep data.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar values 
        infos = self.locals["infos"]
        F = []
        n = []
        Tr = []
        t = []
        r = []
        d_inline = []
        d_inline_phi = []
        d_pnr = []
        d_pnr_phi = []

        for i in range(len(infos)):

            F.append(infos[i]['F'])
            Tr.append(infos[i]['Tr'])
            n.append(infos[i]['n'])

            t.append(infos[i]['t'])
            r.append(infos[i]['r'])

            d_inline.append(infos[i]['d-inline'])
            d_inline_phi.append(infos[i]['d-inline-phi'])
            d_pnr.append(infos[i]['d-pnr'])
            d_pnr_phi.append(infos[i]['d-pnr-phi'])

        self.logger.record(f"timestep/avg_transmitivity", np.mean(t))
        self.logger.record(f"timestep/avg_squeezing", np.mean(r))
        self.logger.record(f"timestep/avg_inline_displacement", np.mean(d_inline))
        self.logger.record(f"timestep/avg_inline_displacement_phase", np.mean(d_inline_phi))
        
        self.logger.record(f"timestep/avg_trace", np.mean(Tr))

        self.logger.record(f"timestep/avg_pnr_displacement", np.mean(d_pnr))
        self.logger.record(f"timestep/avg_pnr_displacement_phase", np.mean(d_pnr_phi))

        self.logger.record(f"timestep/avg_fidelity", np.mean(F))
        self.logger.record(f"timestep/avg_PNR", np.mean(n))
        self.logger.record(f"timestep/trace_env0", infos[0]['Tr'])
        self.logger.record(f"timestep/trace_env1", infos[1]['Tr'])
       
        self.logger.dump(self.num_timesteps)

        return(True)

class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}.{extension}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True
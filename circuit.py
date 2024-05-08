import numpy as np
from numpy import pi, sqrt, diagonal, triu_indices, arccos, concatenate, arange, real, imag

from qutip import Qobj, fidelity, destroy, momentum, position, displace

import strawberryfields as sf
from strawberryfields.ops import Sgate, Squeezed, Dgate, BSgate, MeasureFock, DensityMatrix, LossChannel

from gym import Env
from gym import spaces

from typing import Tuple, List, Dict, Union, Any

from utils import plot_state, squeezed_vacuum


class Circuit(Env): # the time-multiplexed optical circuit (the environment)
    
        def __init__(self, env_params: Dict[str, str],  targets: List[np.ndarray],  
                     seed: int = None, evaluate: bool = False) -> None:
        
            if seed != None:
               np.random.seed(seed) # set random seed for env instance
            
            self.dim: int = env_params["hilbert_dimension"] # dimension of hilbert space 
            self.reward_method: str = env_params["reward_func"]
            self.evaluate: bool = evaluate
            self.max_quad: float = 7.5 # maximum quadrature value for Winger plot
            self.quad_range: np.ndarray = arange(-self.max_quad, self.max_quad, 0.1)
            self.exp: float = env_params["penalty_exponent"] # penalty exponent
            self.sqz_mag: float = np.abs(env_params["initial_sqz"])
            self.sqz_angle: float = np.angle(env_params["initial_sqz"])
            self.max_sqz: float = env_params["max_sqz"]
            self.max_disp: float = env_params["max_disp"]
            self.success_prob: float = 1
            self.loss: float = env_params["loss"]
            self.is_lossy: bool = self.loss > 0.0
            self.t: int = 0 # the current time step
            self.T: int = env_params["max_steps"] # max number of iterations/time steps
            self.num_actions: int = env_params["num_actions"]
            self.steps: List[Dict[str, Union[float, int]]] = []
            self.target_states: List[np.ndarray] = targets
            self.num_target_states = len(self.target_states)

            # get initial state
            self.initial: np.ndarray = squeezed_vacuum(self.sqz_mag, self.sqz_angle, self.dim)
            self.dm: np.ndarray = self.initial
            
            # state space
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=( self.dim**2, ), dtype=np.float32) 
            
            # action space
            minAction = [-1.0] * self.num_actions
            maxAction = [1.0] * self.num_actions
                
            self.action_space = spaces.Box(low=np.array(minAction).astype(np.float32),\
                                           high=np.array(maxAction).astype(np.float32), dtype=np.float32)
        
        def step(action: List, postselect: int=None) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Union[float, int]]]:

            return

        def render(self, name: str, filename: str, is_target: bool=True) -> None:
            # plots the Wigner function and photon number distribution
            if is_target:
               for i in range(self.num_target_states):
                   plot_state(dm=self.target_states[i], max_quad=self.max_quad, points=100, dim=self.dim,\
                              name='target state '+str(i+1), filename=filename+str(i+1))

            else:
                plot_state(dm=self.dm, max_quad=self.max_quad, points=100, dim=self.dim,\
                           name=name, filename=filename)
            
            return
        
        
        def reset(self, seed=1) -> Tuple[np.ndarray, Dict[Any, Any]]:
            # reset the environment
            self.t = 0
            self.steps = []
            self.dm = self.initial
            state: np.ndarray = self.dm[triu_indices(self.dim, k=1)]
            diag: np.ndarray = diagonal(self.dm)
            state: np.ndarray = np.concatenate((real(state), imag(state)), dtype=np.float32, axis=None)
            state: np.ndarray = np.concatenate((state, real(diag)), dtype=np.float32, axis=None)

            return(state, {})
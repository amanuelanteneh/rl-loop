import numpy as np
from numpy import pi, sqrt, diagonal, triu_indices, arccos, concatenate

from qutip import Qobj, fidelity, destroy, momentum, position, displace

import strawberryfields as sf
from strawberryfields.ops import Sgate, Squeezed, Dgate, BSgate, MeasureFock, DensityMatrix, LossChannel

from gym import Env
from gymn import spaces

from typing import Tuple, List, Dict, Union

from utils import plot_state, get_target_state


class Circuit(Env): # the time-multiplexed optical circuit (the enviornment)
    
        def __init__(self, dim: int, initial_sqz: float,\
                 max_timesteps: int, exp: float, sqz_max: float, disp_max: float,\
                 target: str, reward: str, seed: int = None,\ 
                 evaluate: bool = False, loss: float = 0.0) -> None:
        
        if seed != None:
           np.random.seed(seed) # set random seed for env instance
        self.dim = dim # dimension of hilbert space 
        self.reward_method: str = reward
        self.evaluate: bool = evaluate
        self.max_quad: float = 7.5 # maximum quadrature value for Winger plot
        self.quad_range = np.arange(-self.max_quad, self.max_quad, 0.1)
        self.exp: float = exp # penalty exponenet
        self.sqz_mag: float = np.abs(initial_sqz)
        self.sqz_angle: float = np.angle(initial_sqz)
        self.sqz_max: float = sqz_max
        self.disp_max: float = disp_max
        self.success_prob: float = 1
        self.loss: float = loss
        self.is_lossy: bool = loss > 0.0
        self.t: int = 0 # the current time step
        self.T: int = max_timesteps # max number of iterations/time steps
        self.steps: List[Dict[str, Union[float, int]]] = []
        
        # get target states
        state = target.split('|')[0]
        self.target_states: List[np.ndarray] = get_target_states(state, dim, cube, r, D, x)
        self.num_target_states = len(self.target_states)
        
        # create initial state
        prog = sf.Program(2) # create 2-mode circuit
        eng = sf.Engine("fock", backend_options={"cutoff_dim": self.dim})
        self.initial = Squeezed(r=self.sqz_mag, p=self.sqz_angle)
        with prog.context as q: #create inital squeezed state
            self.initial | q[0]
        
        result = eng.run(prog)
        self.psi = result.state
        self.dm = self.psi.reduced_dm([0]) # reduced density matrix
        
        # state space
        self.observation_space = spaces.Box(low=-1, high=1, shape=( self.dim**2, ), dtype=np.float32) 
        
        # action space
        minAction = [-1.0, -1.0, -1.0, -1.0]
        maxAction = [1.0, 1.0, 1.0, 1.0]
            
        self.action_space = spaces.Box(low=np.array(minAction).astype(np.float32),\
                                       high=np.array(maxAction).astype(np.float32), dtype=np.float32)
        
        
        def render(self, name, filename, target=True, steps=None) -> None:
            # plot the Wigner function and photon number distribution
            if target:
               for i in range(self.num_target_states):
                   plotState(dm=self.target_states[i], Qmax=self.max_quad, points=100, dim=self.dim,\
                            name='target-'+str(i+1), fid=-1, filename=filename+str(i+1), step=steps)

            else:
                F = -1
                plotState(dm=self.dm, Qmax=self.Qmax, points=100, numcut=self.dim,\
                           name=name, fid=F, filename=filename, step=steps)
            return
        
        
        def reset(self, seed=1) -> Tuple[np.ndarray, Dict[Any, Any]]:
            # reset the environment
            self.t = 0
            self.steps = []
            eng = sf.Engine("fock", backend_options={"cutoff_dim": self.dim})
            prog = sf.Program(2)
            self.successProb = 1
            with prog.context as q: #create inital squeezed state
                self.initial | q[0]

            result = eng.run(prog)
            self.psi = result.state  
            self.dm = self.psi.reduced_dm([0])

            state: np.ndarray = self.dm[triu_indices(self.dim, k=1)]
            diag: np.ndarray = diagonal(self.dm)
            state = np.concatenate((np.real(state), np.imag(state)), dtype=np.float32, axis=None)
            state = np.concatenate((state, np.real(diag)), dtype=np.float32, axis=None)

            return(state, {})
import numpy as np
from numpy import pi, diagonal, triu_indices, arccos, concatenate, arange, real, imag, sqrt, trace

from scipy.linalg import sinm, cosm

import strawberryfields as sf
from strawberryfields.ops import Squeezed, Dgate, BSgate, MeasureFock, DensityMatrix, LossChannel, CZgate

from qutip import momentum, position

from gym import Env
from gym import spaces

from typing import Tuple, List, Dict, Union

from utils import plot_state, squeezed_vacuum


CIRCUIT_TYPES = { "s~bs": 2, # squeezed state input with no angle control and beamsplitter with only \tau tunable 
                  "s~d~bs": 3, # same as above except with in-loop/in-line displacement with fixed angle of pi/2
                  "s~bs~d": 3, # same as first but with displacement prior to PNR detector with fixed angle of pi/2
                  "s~bs~d-angle": 4, # same as above but displacement angle is now tunable
                  "s~d~bs~d": 4, # same as first but with inline AND PNR displacement both with fixed angle (pi/2)
                  "cz~d-angle": 1 # star cluster state, cz-gate between input and a p-squeezed state with tunable pnr angle only
                }


class Circuit(Env): # the time-multiplexed optical circuit (the environment)
    
        def __init__(self, env_params: Dict[str, Union[int,float,str, bool]],  targets: List[np.ndarray],  
                     seed: int = None, evaluate: bool = False) -> None:
        
            if seed != None:
               np.random.seed(seed) # set random seed for env instance
            
            self.dim = env_params["hilbert_dimension"] # dimension of hilbert space 
            self.reward_method = env_params["reward_func"]
            self.evaluate  = evaluate
            self.max_quad = 5.5 # maximum quadrature value for Winger plot
            self.quad_range: np.ndarray = arange(-self.max_quad, self.max_quad, 0.1)
            self.exp = env_params["penalty_exponent"] # penalty exponent
            self.sqz_mag = np.abs(env_params["initial_sqz"])
            self.sqz_angle = np.angle(env_params["initial_sqz"])
            self.max_sqz  = env_params["max_sqz"]
            self.max_disp = env_params["max_disp"]
            self.pnr_disp = env_params["pnr_disp"]
            self.success_prob = 1.0
            self.loss = env_params["loss"]
            self.is_lossy = self.loss > 0.0
            self.t = 0 # the current time step
            self.T = env_params["max_steps"] # max number of iterations/time steps
            self.circuit_type = env_params["circuit_type"]
            self.num_actions = CIRCUIT_TYPES[self.circuit_type]
            self.state_type = env_params["state_type"]
            self.steps: List[Dict[str, Union[float, int]]] = []
            self.target_states = targets
            self.num_target_states = len(self.target_states)
            self.trace = 1.0
            self.prog = None
            self.prog = sf.Program(2)
            self.eng = sf.Engine("fock", backend_options={"cutoff_dim": self.dim})
            self.cz = None

            if "cz" in self.circuit_type.split("~"):
                self.cz = CZgate(1)

            if self.reward_method == "gkp":
                x = np.array( position(self.dim) )
                p = np.array( momentum(self.dim) )
                
                # create (qubit state) GKP squeezing operators
                # Q0 and Q1 defined in: PhysRevLett.132.210601
                sin_mat1 = sinm(x * (sqrt(pi) / 2 )) 
                sin_mat2 = sinm(p * sqrt(pi) )
                self.Q0 = 2 * ( sin_mat1 @ sin_mat1 ) + 2 * ( sin_mat2 @ sin_mat2 )

                # cos_mat1 = cosm(x * (sqrt(pi) / 2 )) 
                # sin_mat2 = sinm(p * sqrt(pi) )
                # self.Q1 = 2 * ( cos_mat1 @ cos_mat1 ) + 2 * ( sin_mat2 @ sin_mat2 )
            if self.reward_method == "sqr-gkp":
                x = np.array( position(self.dim) )
                p = np.array( momentum(self.dim) )
                
                # create (square grid) GKP squeezing operators
                # Q0 and Q1 defined in: PhysRevLett.132.210601
                sin_mat1 = sinm(x * sqrt(pi / 2) ) 
                sin_mat2 = sinm(p * sqrt(pi / 2) ) 
                self.Q0 = 2 * ( sin_mat1 @ sin_mat1 ) + 2 * ( sin_mat2 @ sin_mat2 )

                # cos_mat1 = cosm(x * sqrt(pi / 2) ) 
                # sin_mat2 = sinm(p * sqrt(pi / 2) )
                # self.Q1 = 2 * ( cos_mat1 @ cos_mat1 ) + 2 * ( sin_mat2 @ sin_mat2 )

            # get initial state
            self.initial = squeezed_vacuum(self.sqz_mag, self.sqz_angle, self.dim)
            self.dm: np.ndarray = self.initial
            
            # state space
            if self.state_type == "dm":
                self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=( self.dim**2, ), dtype=np.float32) 
            elif self.state_type == "pnr":
                self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=( 1 + self.num_actions, ), dtype=np.float32) 
            else:
                raise NotImplementedError("State type not implemented!")
            
            minAction = [-1.0] * self.num_actions
            maxAction = [1.0] * self.num_actions
            
            # action space
            self.action_space = spaces.Box(low=np.array(minAction).astype(np.float32),\
                                           high=np.array(maxAction).astype(np.float32), dtype=np.float32)
        
        def step(self, action: List[float], 
                 postselect: int = None) -> Tuple[np.ndarray, float, bool, Dict[str, Union[float, int]]]:

            self.prog = sf.Program(2) # create 2 mode circuit

            # perform unitary evolution
            transmittivity, r, d_pnr, d_pnr_phi, d_inline, d_inline_phi = self.apply_unitary(action)

            result = self.eng.run(self.prog) 
            
            if self.evaluate:
                dm1 = result.state.reduced_dm([0]) # density matrix of mode 1 before pnr projection
            self.trace = result.state.trace() # get trace of simulation *before* PNR
            
            # measurement (non-unitary)
            measure_prog = sf.Program(self.prog) 
            with measure_prog.context as q:
                if self.is_lossy:
                    LossChannel(1 - self.loss) | q[0]
                MeasureFock(select=postselect) | q[0] # postselected or random PNR measurement on mode 1

            result = self.eng.run(measure_prog)
            n = result.samples[0][0] # the number of detected photons

            if self.evaluate: # if in evaluation mode
                Pn = real(dm1[n][n])
                if round(transmittivity, 2) == 1.0: # if agent resets loop
                    self.success_prob = 1
                    Pn = 1
                elif round(transmittivity, 2) == 0.0: # if agent turns off BS
                    Pn = 1
                self.success_prob *= Pn
            
            self.t += 1 # increment time step 
            self.dm = result.state.reduced_dm([1]) # partial trace over mode 1   
            reward, F = self.get_reward()
            done = self.t == self.T
            # fidelity threshold since agent can't stop loop on its own
            if self.cz != None: 
               done = F >= 0.95 or self.t == self.T

            info = {"Timestep": self.t,
                    "Tr": self.trace,
                    "P": self.success_prob,
                    "F": F,
                    "n": int(n),
                    "t": transmittivity,
                    "r": r,
                    "d-pnr": d_pnr,
                    "d-pnr-phi": d_pnr_phi,
                    "d-inline": d_inline,
                    "d-inline-phi": d_inline_phi,
                     }

            self.steps.append(info)
            
            if self.state_type == "dm":
               state = self.dm[triu_indices(self.dim, k=1)] # get values above diagonal since dm is hermitian
               diag = diagonal(self.dm) # also get diagonal
               state = concatenate((real(state), imag(state)), dtype=np.float32, axis=None)
               state = concatenate((state, real(diag)), dtype=np.float32, axis=None)
            elif self.state_type == "pnr":
                state = np.array( [n / self.dim]  + list(action) ) # concat arrays
            else:
                raise NotImplementedError("State type not implemented!")
            
            return(state, reward, done, info)

        def apply_unitary(self, action: List[float]) -> Tuple[float, float, float, float, float, float]:
            transmittivity = r = 0.0
            d_inline = d_inline_phi = d_pnr = d_pnr_phi = 0.0
            
            if self.circuit_type == "s~bs":
                transmittivity = (action[0] + 1.0)/2.0 # rescale range from [-1,1] -> [0, 1] 
                theta = arccos(transmittivity)
                r = self.max_sqz * action[1]

                with self.prog.context as q:
                    Squeezed(r, 0) | q[1] 
                    DensityMatrix(self.dm) | q[0] # set mode 1 to be output state from prev. step
                    BSgate(theta, 0) | (q[0], q[1])
            
            elif self.circuit_type == "s~d~bs":
                transmittivity = (action[0] + 1.0)/2.0 
                theta = arccos(transmittivity)
                r = self.max_sqz * action[1]
                d_inline = self.max_disp * action[2]
                d_pnr = self.pnr_disp
                d_inline_phi = d_pnr_phi = pi/2

                with self.prog.context as q:
                    Squeezed(r, 0) | q[1] 
                    DensityMatrix(self.dm) | q[0]
                    Dgate(d_inline, d_inline_phi) | q[0]
                    BSgate(theta, 0) | (q[0], q[1])
                    Dgate(d_pnr, d_pnr_phi) | q[0]
            
            elif self.circuit_type == "s~bs~d":
                transmittivity = (action[0] + 1.0)/2.0 
                theta = arccos(transmittivity)
                r = self.max_sqz * action[1]
                d_pnr = self.pnr_disp * action[2]
                d_pnr_phi = pi/2

                with self.prog.context as q:
                    Squeezed(r, 0) | q[1] 
                    DensityMatrix(self.dm) | q[0]
                    BSgate(theta, 0) | (q[0], q[1])
                    Dgate(d_pnr, d_pnr_phi) | q[0]
            
            elif self.circuit_type == "s~bs~d-angle":
                transmittivity = (action[0] + 1.0)/2.0 
                theta = arccos(transmittivity)
                r = self.max_sqz * action[1]
                d_pnr = self.pnr_disp * action[2]
                d_pnr_phi = pi*(action[3] + 1) # [-1,1] -> [0, 2*pi]

                with self.prog.context as q:
                    Squeezed(r, 0) | q[1] 
                    DensityMatrix(self.dm) | q[0]
                    BSgate(theta, 0) | (q[0], q[1])
                    Dgate(d_pnr, d_pnr_phi) | q[0]

            elif self.circuit_type == "s~d~bs~d":
                transmittivity = (action[0] + 1.0)/2.0 
                theta = arccos(transmittivity)
                r = self.max_sqz * action[1]
                d_pnr = self.pnr_disp * action[2]
                d_inline = self.max_disp * action[3]
                d_pnr_phi = d_inline_phi = pi/2

                with self.prog.context as q:
                    Squeezed(r, 0) | q[1] 
                    DensityMatrix(self.dm) | q[0]
                    Dgate(d_inline, d_inline_phi) | q[0]
                    BSgate(theta, 0) | (q[0], q[1])
                    Dgate(d_pnr, d_pnr_phi) | q[0]
            
            elif self.circuit_type == "cz~d-angle":
                d_pnr = self.max_disp
                d_pnr_phi = pi*(action[0] + 1)

                with self.prog.context as q:
                    DensityMatrix(self.initial) | q[1] 
                    DensityMatrix(self.dm) | q[0]
                    self.cz | (q[0], q[1])
                    Dgate(d_pnr, d_pnr_phi) | q[0]
            
            else:
                raise NotImplementedError("Circuit type not implemented!")
            
            return transmittivity, r, d_pnr, d_pnr_phi, d_inline, d_inline_phi

        def get_reward(self) -> Tuple[float, float]:
            if self.reward_method == "fidelity":
                # since we know at least one of the states will be pure (the target state) 
                # we can use this much simpler formula for density matrix fidelity
                F = max( [real( trace( np.array(target) @ self.dm) ) for target in self.target_states] )
                reward = (self.trace**(self.exp/10.0)) * (F**self.exp)       
            
            elif self.reward_method == "gkp" or self.reward_method == "sqr-gkp":
                #xis = [ real( trace(self.dm @ self.Q0) ), real( trace(self.dm @ self.Q1) ) ]
                #min_xi = min(xis)
                xi = real( trace(self.dm @ self.Q0) )
                F = (5 - xi) / 5
                reward = F**self.exp
                

            else:
                raise NotImplementedError("Reward method not implemented!")

            return reward, F

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
        
        
        def reset(self) -> np.ndarray:
            # resets the environment
            self.t = 0
            self.steps = []
            self.eng = sf.Engine("fock", backend_options={"cutoff_dim": self.dim})
            self.prog = sf.Program(2)
            self.dm = self.initial
            self.success_prob = 1.0

            if self.state_type == "dm":
                state = self.dm[triu_indices(self.dim, k=1)]
                diag = diagonal(self.dm)
                state = concatenate((real(state), imag(state)), dtype=np.float32, axis=None)
                state = concatenate((state, real(diag)), dtype=np.float32, axis=None)
            elif self.state_type == "pnr":
                state = np.array( [0.0] * (1+self.num_actions) )
            else:
                raise NotImplementedError("State type not implemented!")

            return(state)
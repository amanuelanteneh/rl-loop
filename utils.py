import numpy as np
from numpy import pi, sqrt, cosh, exp, tanh, outer, arange, meshgrid, real, diag

from qutip import wigner, squeeze, displace, momentum, position, fock, Qobj, coherent

import matplotlib.pyplot as plt
from matplotlib import cm

import math

from typing import List, Union


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


def episode_callback_single() -> None:
    return

def timestep_callback_single() -> None:
    return

def episode_callback_multi() -> None:
    return

def timestep_callback_multi() -> None:
    return

def checkpoint_callback() -> None:
    return
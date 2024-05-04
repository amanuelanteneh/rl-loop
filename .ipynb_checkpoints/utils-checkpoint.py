from typing import Callable, Dict
from gymnasium import Env
from circuit import Circuit


def make_env(env_parameters: Dict[str, str], rank: int, seed: int) -> Callable:
    """
    Utility function for multiprocessed env
    
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> Env:
        env = Circuit(dim, intialSqz, maxSteps, exp, sqzMax, dispMax, pnr_disp, tune_pnr_phi, target, \
                      reward, seed+rank, False, loss)
       
        return env

    return _init
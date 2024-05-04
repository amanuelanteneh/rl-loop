import numpy as np
from numpy import pi, sqrt, diagonal, triu_indices, arccos, concatenate

from qutip import Qobj, fidelity, destroy, momentum, position, displace

import strawberryfields as sf
from strawberryfields.ops import Sgate, Squeezed, Dgate, BSgate, MeasureFock, DensityMatrix, LossChannel

from gymnasium import Env
from gymnasium import spaces

from typing import Tuple, List

from utils import plot_state, get_target_state
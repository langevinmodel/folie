from . import _version

__version__ = _version.get_versions()["version"]

import os

os.environ["SPARSE_AUTO_DENSIFY"] = "1"


from .data import Trajectories, Trajectory

from .domains import *

from .estimation import *
from .models import *

from .simulations import Simulator

from . import functions

from . import analysis

from . import _version

from . import fem

__version__ = _version.get_versions()["version"]

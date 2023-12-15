from . import _version

__version__ = _version.get_versions()["version"]


from .data import Trajectories

from .estimation import *
from .models import *

from . import function_basis

from .simulations import Simulator

from . import functions

from . import _version

__version__ = _version.get_versions()["version"]

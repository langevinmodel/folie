import os

_which_numpy = os.environ.get("FOLIE_NUMPY", "numpy")

if _which_numpy == "cupy":
    import cupy as np
elif _which_numpy == "jax":
    import jax.numpy as np
else:
    import numpy as np

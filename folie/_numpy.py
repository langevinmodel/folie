import os

_which_numpy = os.environ.get("FOLIE_NUMPY", "numpy")

if _which_numpy.lower() == "cupy":
    import cupy as np
elif _which_numpy.lower() == "jax":
    import jax.numpy as np
elif _which_numpy.lower() == "cunumeric":
    import cunumeric as np
else:
    import numpy as np

from collections.abc import MutableSequence, Mapping
from .._numpy import np
from ._data_statistics import traj_stats, sum_stats, representative_array, histogram, hexbin


def Trajectory(dt, x, v=None, bias=None):
    """
    Create dict_like object that encaspulate the trajectory data
    TODO: Use xarray DataSet?
    """
    trj = {"x": np.atleast_2d(x), "dt": dt}
    if v is not None:
        trj["v"] = np.atleast_2d(v)
    if bias is not None:
        trj["bias"] = np.atleast_2d(bias)
    return trj


class Trajectories(MutableSequence):
    """
    Set of trajectories
    """

    def __init__(self, dt=None):
        self.dt = dt
        self.trajectories_data = []
        self.dim = None
        self.stats_data = None

    def _check_data(self, v):
        """ """
        if not isinstance(v, Mapping):
            v = Trajectory(self.dt, v)
        if len(v["x"].shape) == 1:
            dim_x = 1
            v["x"] = v["x"].reshape(-1, 1)
        else:
            dim_x = v["x"].shape[-1]
        if self.dim is None:
            self.dim = dim_x
        elif self.dim != dim_x:
            raise ValueError("Inconsitent dimension between previously stored trajectory and currently added trajectory")
        if self.dt is None:
            self.dt = v["dt"]
        return v

    def __len__(self):
        return len(self.trajectories_data)

    def __getitem__(self, i):
        return self.trajectories_data[i]

    def __delitem__(self, i):
        del self.trajectories_data[i]

    def __setitem__(self, i, v):
        self.trajectories_data[i] = self._check_data(v)

    def insert(self, i, v):
        self.trajectories_data.insert(i, self._check_data(v))

    def __str__(self):
        return "".join(["Trajectory of length {} and dimension {}.\n".format(len(trj["x"]), self.dim) for trj in self.trajectories_data])

    def remove_key(self, key):
        """
        Safely removes a specific key from all trajectories in the dataset.

        Parameters
        ----------
        key : str
            The dictionary key to remove (e.g., 'density', 'bias').

        Returns
        -------
        self : Trajectories
            Reference to self to allow method chaining.
        """
        for trj in self.trajectories_data:
            if key in trj:
                del trj[key]
        return self

    def representative_array(self, Npoints=75, **kwargs):
        return representative_array(self.stats, Npoints, **kwargs)

    @property
    def stats(self):
        """
        Basic statistics on the data
        """
        if self.stats_data is None:
            self.stats_data = traj_stats(self.trajectories_data[0]["x"])
            for trj in self.trajectories_data[1:]:
                self.stats_data = sum_stats(self.stats_data, traj_stats(trj["x"]))
        return self.stats_data

    @property
    def nobs(self):
        return np.sum([trj["x"].shape[0] for trj in self.trajectories_data])

    @property
    def weights(self):
        return np.array([trj["x"].shape[0] for trj in self.trajectories_data])

    def to_xarray(self):
        """
        Return the data as a list of Dataset where extra variables are denoted by their name.
        The time step 'dt' is stored as a global attribute in each Dataset.
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("The 'xarray' package is required to use to_xarray().")

        datasets = []
        for trj in self.trajectories_data:
            data_vars = {}
            for key, val in trj.items():
                if key == "dt":
                    continue  # dt is managed globally/as an attribute

                val_arr = np.asarray(val)
                # Assign dimension names intelligently based on array shape
                if val_arr.ndim == 0:
                    dims = []
                elif val_arr.ndim == 1:
                    dims = ["time"]
                elif val_arr.ndim == 2:
                    dims = ["time", "dim"]
                else:
                    dims = ["time"] + [f"dim_{i}" for i in range(1, val_arr.ndim)]

                data_vars[key] = (dims, val_arr)

            # Create the dataset and attach the dt attribute
            ds = xr.Dataset(data_vars, attrs={"dt": self.dt})
            datasets.append(ds)

        return datasets

    @classmethod
    def from_xarray(cls, traj_list, data_key="x"):
        """
        Take as input a list of xarray Dataset and reconstruct the Trajectories object.
        """
        if not traj_list:
            return cls()

        # Try to infer dt from the first dataset's attributes
        dt = traj_list[0].attrs.get("dt", None)
        obj = cls(dt=dt)

        for ds in traj_list:
            trj_dict = {}

            # Extract all data variables as numpy arrays
            for var_name in ds.data_vars:
                trj_dict[var_name] = ds[var_name].to_numpy()

            # Ensure the primary data_key exists (typically "x")
            if data_key not in trj_dict:
                raise ValueError(f"Expected key '{data_key}' not found in the xarray Dataset variables.")

            obj.append(trj_dict)

        return obj

    def fit_density(self, method="hist", **kwargs):
        """
        Fits the density object on the current trajectory data and stores it as a class member.
        """
        X_all = np.concatenate([trj["x"] for trj in self.trajectories_data], axis=0)
        self.density_method = method
        self.density_kwargs = kwargs

        if method == "hist":
            bins = kwargs.get("bins", 50)
            data_range = kwargs.get("data_range", None)

            # 1. Call external histogram function
            H, edges = histogram(self, bins=bins, data_range=data_range)
            self.density_obj = {"H": H, "edges": edges}

        elif method == "hexbin":
            gridsize = kwargs.get("gridsize", 50)

            # 2. Call external hexbin function
            x_centers, y_centers, C_mask, extent = hexbin(self, gridsize=gridsize)

            # Build a KDTree of the hex centers so we can quickly evaluate arbitrary new points
            from sklearn.neighbors import NearestNeighbors

            centers = np.column_stack([x_centers, y_centers])
            nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(centers)

            # Store hex_size to assign 0 to points way outside the grid
            xmin, xmax = extent[0], extent[1]
            hex_size = (xmax - xmin) / gridsize

            self.density_obj = {"nbrs": nbrs, "C": C_mask, "hex_size": hex_size}

        elif method == "knn":
            from sklearn.neighbors import NearestNeighbors

            k = kwargs.get("k", min(50, len(X_all)))
            nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1).fit(X_all)
            self.density_obj = {"nbrs": nbrs, "k": k, "dim": self.dim}

        elif method == "kde":
            from sklearn.neighbors import KernelDensity

            sigma = np.std(X_all, axis=0).mean()
            bw = kwargs.get("bandwidth", (sigma if sigma > 0 else 1.0) * (len(X_all) ** (-1.0 / (self.dim + 4))))
            kde = KernelDensity(bandwidth=bw).fit(X_all)
            self.density_obj = {"kde": kde}

        else:
            raise ValueError(f"Unknown density method: {method}")

        # Compute the global mean density of the training set to normalize future evaluations
        raw_densities = self.compute_density(X_all, force_eval=True, normalize=False)
        self.density_mean_ = np.mean(raw_densities) + 1e-10

        return self

    def compute_density(self, x=None, method=None, force=False, normalize=True, **kwargs):
        """
        Compute density at given points `x`.
        If `x` is None, computes for all internal trajectories and attaches to trj["density"].
        Calls `fit_density` if the density object doesn't exist or parameters changed.
        """
        # 1. Determine if we need to re-fit the density object
        target_method = method or self.density_method or "hist"
        need_fit = force or self.density_obj is None or self.density_method != target_method or self.density_kwargs != kwargs

        if need_fit and not kwargs.get("force_eval", False):
            self.fit_density(method=target_method, **kwargs)

        # 2. Internal evaluator function for arbitrary points
        def evaluate(points):
            if self.density_method == "hist":
                H, edges = self.density_obj["H"], self.density_obj["edges"]
                dim = points.shape[1]
                bin_indices = tuple(np.clip(np.digitize(points[:, i], edges[i][:-1]) - 1, 0, H.shape[i] - 1) for i in range(dim))
                return H[bin_indices]

            elif self.density_method == "hexbin":
                nbrs, C, hex_size = self.density_obj["nbrs"], self.density_obj["C"], self.density_obj["hex_size"]
                distances, indices = nbrs.kneighbors(points)
                densities = C[indices.ravel()]
                # Points further than hex_size from a center are outside the masked grid -> set to 0
                densities[distances.ravel() > hex_size] = 0.0
                return densities

            elif self.density_method == "knn":
                nbrs, dim = self.density_obj["nbrs"], self.density_obj["dim"]
                distances, _ = nbrs.kneighbors(points)
                return 1.0 / (distances[:, -1] ** dim + 1e-10)

            elif self.density_method == "kde":
                kde = self.density_obj["kde"]
                return np.exp(kde.score_samples(points))

        # 3. Evaluate and return
        if x is not None:
            # User provided arbitrary points to evaluate
            densities = evaluate(np.asarray(x))
            return (densities / self.density_mean_) if normalize else densities
        else:
            # Apply to all internal trajectories
            idx = 0
            all_points = np.concatenate([trj["x"] for trj in self.trajectories_data], axis=0)
            all_densities = evaluate(all_points)
            if normalize:
                all_densities /= self.density_mean_

            for trj in self.trajectories_data:
                n = len(trj["x"])
                trj["density"] = all_densities[idx : idx + n]
                idx += n
            return self

    def grid(self):
        """
        Return the grid centers from the fitted histogram density.

        Returns
        -------
        centers : list of ndarray
            Bin centers for each dimension.
        """
        if self.density_obj is None or "edges" not in self.density_obj:
            raise RuntimeError("No density fitted. Call fit_density() first.")
        edges = self.density_obj["edges"]
        return [(e[:-1] + e[1:]) / 2 for e in edges]

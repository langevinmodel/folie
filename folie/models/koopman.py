from .._numpy import np

from ..base import Model


class KoopmanModel(Model):

    def __init__(self, dim=1, basis=False, **kwargs):
        r"""
        Koopman evolution operator :footcite:`Thiede2019`

        References
        --------------

        """

        self._dim = dim
        self.is_biased = False
        self.basis = basis

    @property
    def dim(self):
        """
        Dimensionnality of the model
        """
        return self._dim

    @dim.setter
    def dim(self, dim):
        """
        Dimensionnality of the model
        """
        if dim == 0:
            dim = 1
        if dim != self._dim:
            raise ValueError("Dimension did not match dimension of the model. Change model or review dimension of your data")

    @property
    def coefficients(self):
        """Access the coefficients"""
        return self.basis.coefficients

    @coefficients.setter
    def coefficients(self, vals):
        """Set parameters, used by fitter to move through param space"""
        self.basis.coefficients = vals.ravel()

    def preprocess_traj(self, trj, **kwargs):
        if hasattr(self.basis.domain, "localize_data"):
            # Check if domain are compatible
            cells_idx, loc_x = self.basis.domain.localize_data(trj["x"], **kwargs)
            trj["cells_idx"] = cells_idx
            trj["loc_x"] = loc_x
        return trj

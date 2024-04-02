import skfem
from skfem import BilinearForm
from skfem.helpers import grad, dot, mul

# TODO: write functions that directly give the right form depending of the model


class LangevinBilinearForm:
    """
    A class to compute value of bilinear form
    """

    __name__ = "Langevin"

    def __init__(self, model):
        """"""
        self.model = model


class LangevinOverdamped(LangevinBilinearForm):
    __name__ = "LangevinOverdamped"

    def __call__(self, u, v, w):
        """
        Return generator for overdamped Langevin equation
        """
        dim, nelem, nquadra = w["x"].shape
        X = w["x"].reshape(dim, -1)
        F = self.model.force(X.T).T.reshape(dim, nelem, nquadra)
        D = self.model.diffusion(X.T).T.reshape(dim, dim, nelem, nquadra)  # should be a dim*dim*w["x"].shape

        return -1 * (dot(grad(v), mul(D, grad(u))) - v * dot(F, grad(u)))


def grammian(u, v, w):
    return u * v


def generate_fem(model, mesh, element):
    """
    Compute FEM matrix and solve committor equation
    """
    langevinform = BilinearForm(LangevinOverdamped(model))
    basis = skfem.InteriorBasis(mesh, element)
    A = skfem.asm(langevinform, basis)
    M = skfem.asm(BilinearForm(grammian), basis)
    return A, M, basis

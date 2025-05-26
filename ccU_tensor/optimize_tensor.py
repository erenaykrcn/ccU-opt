import numpy as np
from ansatz_tensor import ansatz_tensor, ansatz_grad_vector_tensor
from hessian_tensor import ansatz_hessian_matrix_tensor
from rqcopt.trust_region import riemannian_trust_region_optimize
from utils_tensor import polar_decomp, real_to_antisymm



def err_tensor(glist, U, L, perms, cU):
    return -np.trace(cU.conj().T @ ansatz_tensor(glist, U, L, perms)).real


def optimize_circuit_tensor(L: int, U, cU, Glist_start, perms, **kwargs):
    """
    Optimize the quantum gates using a trust-region method.
    """

    # target function
    f = lambda glist: err_tensor(glist, U, L, perms, cU)
    gradfunc = lambda glist: -ansatz_grad_vector_tensor(glist, cU, U, L, perms)
    hessfunc = lambda glist: -ansatz_hessian_matrix_tensor(glist, cU, U, L, perms)
    
    # quantify error by spectral norm
    errfunc = lambda glist: np.linalg.norm(ansatz_tensor(glist, U, L, perms) - cU, ord=2)
    kwargs["gfunc"] = errfunc
    # perform optimization
    Glist, f_iter, err_iter = riemannian_trust_region_optimize(
        f, retract_unitary_list, gradfunc, hessfunc, np.stack(Glist_start), **kwargs)
    return Glist, f_iter, err_iter


def retract_unitary_list(vlist, eta):
    """
    Retraction for unitary matrices, with tangent direction represented as anti-symmetric matrices.
    """
    n = len(vlist)
    eta = np.reshape(eta, (n, 4, 4))
    dvlist = [vlist[j] @ real_to_antisymm(eta[j]) for j in range(n)]
    return np.stack([polar_decomp(vlist[j] + dvlist[j])[0] for j in range(n)])


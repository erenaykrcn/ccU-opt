import numpy as np
from ansatz import ansatz, ansatz_grad_vector
#from hessian import ansatz_hessian_matrix
from utils import polar_decomp, real_to_antisymm
from rqcopt.trust_region import riemannian_trust_region_optimize



def err(glist, U, L, perms, cU):
    return -np.trace(cU.conj().T @ ansatz(glist, U, L, perms)).real

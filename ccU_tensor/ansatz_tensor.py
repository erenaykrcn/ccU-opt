import numpy as np
from utils_tensor import (
	applyG_tensor, applyG_block_tensor,
	partial_trace_keep, antisymm_to_real, antisymm, I2, X, Y, Z,
	project_unitary_tangent, real_to_antisymm, partial_trace_keep_tensor
	)


def ansatz_tensor(Glist, U, L, perms):
    """
    Tensor-based ansatz construction, applying gates directly
    to a (2,)*2L tensor without forming large matrices.
    """
    assert len(Glist) == 2 * len(perms)
    Vs, Ws = Glist[:len(perms)], Glist[len(perms):]

    ret_tensor = np.eye(2**L, dtype=complex).reshape([2]*2*L)
    for i, V in enumerate(Vs):
        ret_tensor = applyG_block_tensor(V, ret_tensor, L, perms[i])

    # Apply U global operator
    ret_tensor = ret_tensor.reshape(2**L, 2**L)
    ret_tensor = (U @ ret_tensor).reshape([2]*2*L)
    # Future TODO: Use tensordot operation for U given as tensor.

    for i, W in enumerate(Ws):
        ret_tensor = applyG_block_tensor(W, ret_tensor, L, perms[i])

    return ret_tensor.reshape(2**L, 2**L)


def ansatz_grad_tensor(V, L, U_tilde_tensor, perm):
    G = np.zeros_like(V, dtype=complex)
    for i in range(L // 2):
        k, l = perm[2 * i], perm[2 * i + 1]

        U_working = U_tilde_tensor.copy()
        for j in range(i):
            k_, l_ = perm[2 * j], perm[2 * j + 1]
            U_working = applyG_tensor(V, U_working, k_, l_)

        for j in range(i + 1, L // 2):
            k_, l_ = perm[2 * j], perm[2 * j + 1]
            U_working = applyG_tensor(V, U_working, k_, l_)

        T = partial_trace_keep(U_working.reshape(2**L, 2**L), [k, l], L)

        if k > l:
            SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
            T = SWAP @ T @ SWAP

        G += T
    return G



def ansatz_grad_vector_tensor(Glist, cU, U, L, perms, flatten=True, unprojected=False):
    Vlist = Glist[:len(perms)]
    Wlist = Glist[len(perms):]

    grads_V = []
    for i, V in enumerate(Vlist):
        U_tilde = np.eye(2**L).reshape([2]*2*L)
        for j in range(i+1, len(perms)):
            U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
        U_tilde = (U @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
        for j in range(len(perms)):
            U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])
        U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
        for j in range(i):
            U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
        grads_V.append(ansatz_grad_tensor(V, L, U_tilde, perms[i]).conj().T)

    grads_W = []
    for i, W in enumerate(Wlist):
        U_tilde = np.eye(2**L).reshape([2]*2*L)
        for j in range(i+1, len(perms)):
            U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])

        U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)

        for j in range(len(perms)):
            U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
        U_tilde = (U @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)

        for j in range(i):
            U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])
        grads_W.append(ansatz_grad_tensor(W, L, U_tilde, perms[i]).conj().T)

    grad = np.stack(grads_V + grads_W)

    if unprojected:
        return grad

    # Project onto tangent space.
    if flatten:
        return np.stack([
            antisymm_to_real(antisymm(Glist[j].conj().T @ grad[j]))
            for j in range(len(grad))
        ]).reshape(-1)
    else:
        return np.stack([
            antisymm_to_real(antisymm(Glist[j].conj().T @ grad[j]))
            for j in range(len(grad))
        ])






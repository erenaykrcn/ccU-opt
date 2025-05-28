import numpy as np
from utils_dagger import (
	applyG_tensor, applyG_block_tensor,
	partial_trace_keep, antisymm_to_real, antisymm, I2, X, Y, Z,
	project_unitary_tangent, real_to_antisymm, partial_trace_keep_tensor
	)


def ansatz_dagger(Glist, U, L, perms):
    assert len(Glist) == len(perms)

    ret_tensor = np.eye(2**L, dtype=complex).reshape([2]*2*L)
    for i, G in enumerate(Glist[::-1]):
        ret_tensor = applyG_block_tensor(G.conj().T, ret_tensor, L, perms[len(perms)-1-i])

    # Apply U global operator
    ret_tensor = ret_tensor.reshape(2**L, 2**L)
    ret_tensor = (U @ ret_tensor).reshape([2]*2*L)
    # Future TODO: Use tensordot operation for U given as tensor.

    for i, G in enumerate(Glist):
        ret_tensor = applyG_block_tensor(G, ret_tensor, L, perms[i])

    return ret_tensor.reshape(2**L, 2**L)


def ansatz_grad_dagger(A, B, V, L, perm):
    G = np.zeros_like(V, dtype=complex)

    U_tilde_tensor1 = (A.reshape(2**L, 2**L) @ applyG_block_tensor(V.conj().T, B, L, perm).reshape(2**L, 2**L)).reshape([2]*2*L)
    U_tilde_tensor2 = (B.reshape(2**L, 2**L) @ applyG_block_tensor(V, A, L, perm).reshape(2**L, 2**L) ).reshape([2]*2*L)
    U_tilde_tensors = [U_tilde_tensor1, U_tilde_tensor2]

    for layer in range(2):
        for i in range(L // 2):
            k, l = perm[2 * i], perm[2 * i + 1]

            U_working = U_tilde_tensors[layer].copy()
            for j in range(i):
                k_, l_ = perm[2 * j], perm[2 * j + 1]
                U_working = applyG_tensor(V if layer==0 else V.conj().T, U_working, k_, l_)

            for j in range(i + 1, L // 2):
                k_, l_ = perm[2 * j], perm[2 * j + 1]
                U_working = applyG_tensor(V if layer==0 else V.conj().T, U_working, k_, l_)

            T = partial_trace_keep(U_working.reshape(2**L, 2**L), [k, l], L)

            if k > l:
                SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
                T = SWAP @ T @ SWAP

            G += T if layer==0 else T.conj().T
    return G



def ansatz_grad_vector_dagger(Glist, cU, U, L, perms, flatten=True, unprojected=False):
    eta=len(perms)

    grad = []
    for i, G in enumerate(Glist):
        B = np.eye(2**L).reshape([2]*2*L)
        for j in range(i+1, eta):
            B = applyG_block_tensor(Glist[j], B, L, perms[j])
        B = (cU.conj().T @ B.reshape(2**L, 2**L)).reshape([2]*2*L)
        for j in range(eta-1, i, -1):
            B = applyG_block_tensor(Glist[j].conj().T, B, L, perms[j])

        A = np.eye(2**L).reshape([2]*2*L)
        for j in range(i-1, -1, -1):
            A = applyG_block_tensor(Glist[j].conj().T, A, L, perms[j])
        A = (U @ A.reshape(2**L, 2**L)).reshape([2]*2*L)
        for j in range(i):
            A = applyG_block_tensor(Glist[j], A, L, perms[j])
        grad.append(ansatz_grad_dagger(A, B, Glist[i], L, perms[i]).conj().T)


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






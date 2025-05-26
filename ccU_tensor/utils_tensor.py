import numpy as np
import scipy.sparse as sp 
import qutip


I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


import numpy as np

def applyG_tensor(G, U_tensor, k, l):
    """
        Performs a 'left' multiplication of applying G
        two qubit gate to qubits k and l.
    """
    L = U_tensor.ndim // 2
    assert G.shape == (4, 4), "G must be a 2-qubit gate (4x4)"
    assert 0 <= k < L and 0 <= l < L and k != l

    # Ensure k < l for consistency
    if k > l:
        k, l = l, k
        SWAP = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]])
        G = SWAP @ G @ SWAP

    # Reshape G to tensor form
    G_tensor = G.reshape(2, 2, 2, 2)
    input_axes = [k, l]
    output_axes = [L + i for i in range(L)]

    # Perform contraction: G_{ab,cd} * U_{...a...b..., ...}
    U_tensor = np.tensordot(G_tensor, U_tensor, axes=([2, 3], input_axes))

    # Move axes back into original order
    new_axes = list(range(2))  # G's output legs
    insert_at = input_axes[0]
    remaining_axes = list(range(2 * L))
    for t in sorted(input_axes, reverse=True):
        del remaining_axes[t]
    for i, ax in enumerate(new_axes):
        remaining_axes.insert(insert_at + i, ax)

    U_tensor = np.moveaxis(U_tensor, range(2), input_axes)
    return U_tensor



def applyG_block_tensor(G, U_tensor, L, perm):
    """
    Applies the 2-qubit gate G to every (k, l) in `perm` (length 2n) 
    on the (2,)*2L tensor U_tensor.

    G is a (4, 4) matrix.
    """
    assert len(perm) % 2 == 0
    for j in range(len(perm) // 2):
        k = perm[2 * j]
        l = perm[2 * j + 1]
        U_tensor = applyG_tensor(G, U_tensor, k, l)
    return U_tensor



def partial_trace_keep(U, keep_qubits, N):
	# Convert NumPy matrix to QuTiP Qobj
	full_dim = [2] * N  # Each qubit has dimension 2
	rho = qutip.Qobj(U, dims=[full_dim, full_dim])

	reduced_rho = rho.ptrace(keep_qubits)
	return reduced_rho.full()


def partial_trace_keep_tensor(rho_tensor, keep, L):
    all_idx = list(range(L))
    traced = [q for q in all_idx if q not in keep]
    
    idx = list(range(2*L))
    i_axes = traced
    j_axes = [L + q for q in traced]
    
    return np.trace(rho_tensor, axis1=i_axes, axis2=j_axes).reshape(4, 4)



def antisymm_to_real(w):
	return w.real + w.imag


def antisymm(w):
	return 0.5 * (w - w.conj().T)


def symm(w):
	return 0.5 * (w + w.conj().T)

def project_unitary_tangent(u, z):
    return z - u @ symm(u.conj().T @ z)

def real_to_antisymm(r):
	return 0.5*(r - r.T) + 0.5j*(r + r.T)


def polar_decomp(a):
	u, s, vh = np.linalg.svd(a)
	return u @ vh, (vh.conj().T * s) @ vh


def u4_basis():
    basis = []

    # Off-diagonal generators
    for i in range(4):
        for j in range(i+1, 4):
            # Real symmetric: i(E_ij + E_ji)
            M = np.zeros((4, 4), dtype=complex)
            M[i, j] = 1
            M[j, i] = 1
            basis.append(1j * M)

            # Imaginary antisymmetric: i(-iE_ij + iE_ji)
            M = np.zeros((4, 4), dtype=complex)
            M[i, j] = -1j
            M[j, i] = 1j
            basis.append(1j * M)

    # Diagonal generators (including identity)
    for i in range(4):
        M = np.zeros((4, 4), dtype=complex)
        M[i, i] = 1
        basis.append(1j * M)

    return basis  # 6 (real sym) + 6 (imag antisym) + 4 (diagonal) = 16
















	
		





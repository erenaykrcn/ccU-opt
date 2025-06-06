import numpy as np
import qutip


ket_0 = np.array([[1],[0]])
ket_1 = np.array([[0],[1]])
rho_0_anc = ket_0 @ ket_0.T
rho_1_anc = ket_1 @ ket_1.T

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# TODO: Pre-store the swaps
# TODO: Turn the 'to_ret' of applyG should be sparse


def otimes(matrices):
	mat = np.eye(1)
	for matrix in matrices:
		mat = np.kron(mat, matrix)
	return mat


def swap_matrix(n, q1, q2):
	"""
	Creates a swap matrix for qubits q1 and q2 in an n-qubit system.
	"""

	dim = 2 ** n
	swap = np.eye(dim)
	for i in range(dim):
		binary = list(format(i, f'0{n}b'))  # Convert index to binary
		if binary[q1] != binary[q2]:  # Swap bits if different
			binary[q1], binary[q2] = binary[q2], binary[q1]
			j = int("".join(binary), 2)
			swap[i, i], swap[i, j] = 0, 1  # Swap rows in identity
	return swap
	

def permute_operation(U, k, l, N):
	"""
	Applies swaps to move qubits (k, l) to (1,2),
	applies the unitary U, then swaps them back.
	"""
	if k == 0 and l == 1:
		return U  # Already in the correct position
	# Swap k to position 1
	S1 = swap_matrix(N, 0, k) if k != 0 else np.identity(2 ** N)
	# Swap l to position 2
	S2 = swap_matrix(N, 1, l) if l != 1 else np.identity(2 ** N)
	# Apply swaps before and after U
	return S1 @ S2 @ U @ S2 @ S1


def applyG(G, k, l, N):
	"""
		Takes the 2-qubit gate G and applies it to
		qubits k and l.
	"""

	# !ANOTHER FIX:
	# Ensure k < l for consistent behavior
	if k > l:
		k, l = l, k
		# Reverse the qubit order inside G
		SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
		G = SWAP @ G @ SWAP

	G_01 = otimes([G] + [I2 for i in range(N-2)])
	to_ret = G_01
	to_ret = permute_operation(to_ret, k, l, N) if (k!=0 or l!=1) else to_ret
	return to_ret


def applyG_block(G, L, perm):
	# Applies G_i (V_i or W_i) to every qubit of the given permutation.
	U = np.eye(2**L)
	for j in range(len(perm)//2):
		U = applyG(G, perm[2*j], perm[2*j+1], L) @ U
	return U


def partial_trace_keep(U, keep_qubits, N):
	# Convert NumPy matrix to QuTiP Qobj
	full_dim = [2] * N  # Each qubit has dimension 2
	rho = qutip.Qobj(U, dims=[full_dim, full_dim])

	reduced_rho = rho.ptrace(keep_qubits)
	return reduced_rho.full()


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
















	
		





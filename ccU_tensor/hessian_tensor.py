import numpy as np
from ansatz_tensor import ansatz_grad_tensor, ansatz_grad_vector_tensor
from utils_tensor import (
	applyG_tensor, applyG_block_tensor,
	partial_trace_keep, antisymm_to_real, antisymm,
	project_unitary_tangent, real_to_antisymm
	)


def ansatz_hess_single_layer_tensor(V, L, Z, U_tilde_, perm):
	"""
		Computes single permutation hessian for the i==k.
	"""
	G = np.zeros_like(V, dtype=complex)
	for i in range(L//2):
		
		k, l = (perm[2*i], perm[2*i+1])
		for z in range(L//2):
			U_tilde = U_tilde_.copy()
			if z==i:
				continue

			for j in range(i):
				k_, l_ = (perm[2*j], perm[2*j+1])
				if z==j:
					U_tilde = applyG_tensor(Z, U_tilde, k_, l_)
				else:
					U_tilde = applyG_tensor(V, U_tilde, k_, l_)

			for j in list(range(i+1, L//2))[::-1]:
				k_, l_ = (perm[2*j], perm[2*j+1])
				if z==j:
					U_tilde = applyG_tensor(Z, U_tilde, k_, l_) # Reversed left to right multiplication here!
				else:
					U_tilde = applyG_tensor(V, U_tilde, k_, l_) # Reversed left to right multiplication here!

			# Take partial trace wrt all qubits but k, l.
			T = partial_trace_keep(U_tilde.reshape(2**L, 2**L), [k, l], L)
			if k>l:
				# Partial trace interprets the qubit l as the first qubit of 
				# the resulting two qubit gate. We need to fix that.
				SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
				T = SWAP @ T @ SWAP
			G += T
	return G


def ansatz_grad_directed_tensor(V, U_tilde, L, Z, perm):
	assert V.shape == (4, 4)
	assert Z.shape == (4, 4)
	assert L % 2 == 0

	G = np.zeros((2**L, 2**L), dtype=complex).reshape([2]*2*L)
	for i in range(L//2):
		U = np.eye(2**L).reshape([2]*2*L)
		k, l = (perm[2*i], perm[2*i+1])

		for j in range(i):
			k_, l_ = (perm[2*j], perm[2*j+1])
			U = applyG_tensor(V, U, k_, l_) 

		U = applyG_tensor(Z, U, k, l)

		for j in list(range(i+1, L//2))[::-1]:
			k_, l_ = (perm[2*j], perm[2*j+1])
			U = applyG_tensor(V, U, k_, l_)
		G += U
	U_tilde = G.reshape(2**L, 2**L) @ U_tilde.reshape(2**L, 2**L)
	return U_tilde.reshape([2]*2*L)


def ansatz_hess_V(Vlist, Wlist, L, Z, k, cU, U, perms, unprojected=False):
	"""
		Assumes the Z_k grad. evaluation happens in the V gates layer.
		There are 4 possibilities for the i sum: i=k, i<k and i>k within 
		the V layer and i within the W layer. 
	"""
	dVlist = [None for V in Vlist]
	dWlist = [None for W in Wlist]

	# i runs over Vlist, i>k
	for i in range(k+1, len(Vlist)):
		U_tilde = np.eye(2**L).reshape([2]*2*L)
		for j in range(i+1, len(Vlist)):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
		U_tilde = (U @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(len(Wlist)):
			U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])
		U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(k):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
		U_tilde = ansatz_grad_directed_tensor(Vlist[k], U_tilde, L, Z, perms[k])
		for j in range(k+1, i):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
		dVi = ansatz_grad_tensor(Vlist[i], L, U_tilde, perms[i]).conj().T
		dVlist[i] = dVi if unprojected else project_unitary_tangent(Vlist[i], dVi)

	# i runs over Vlist, k>i
	for i in range(k):
		U_tilde = np.eye(2**L).reshape([2]*2*L)
		for j in range(i+1, k):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
		U_tilde = ansatz_grad_directed_tensor(Vlist[k], U_tilde, L, Z, perms[k])
		for j in range(k+1, len(Vlist)):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
		U_tilde = (U @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(len(Wlist)):
			U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])
		U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(i):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])

		dVi = ansatz_grad_tensor(Vlist[i], L, U_tilde, perms[i]).conj().T
		dVlist[i] = dVi if unprojected else project_unitary_tangent(Vlist[i], dVi)

	# i runs within Wlist.
	for i in range(len(Wlist)):
		U_tilde = np.eye(2**(L)).reshape([2]*2*L)
		for j in range(i+1, len(Wlist)):
			U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])
		U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(k):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
		U_tilde = ansatz_grad_directed_tensor(Vlist[k], U_tilde, L, Z, perms[k])
		for j in range(k+1, len(Vlist)):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
		U_tilde = (U @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(i):
			U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])
		
		dWi = ansatz_grad_tensor(Wlist[i], L, U_tilde, perms[i]).conj().T
		dWlist[i] = dWi if unprojected else project_unitary_tangent(Wlist[i], dWi)

	# i=k case.
	i = k
	U_tilde = np.eye(2**L).reshape([2]*2*L)
	for j in range(i+1, len(Vlist)):
		U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
	U_tilde = (U @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
	for j in range(len(Wlist)):
		U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])
	U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
	for j in range(i):
		U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])

	G = ansatz_hess_single_layer_tensor(Vlist[k], L, Z, U_tilde, perms[k]).conj().T

	# Projection.
	if not unprojected:
		V = Vlist[k]
		G = project_unitary_tangent(V, G)
		grad = ansatz_grad_tensor(V, L, U_tilde, perms[k]).conj().T

		G -= 0.5 * (Z @ grad.conj().T @ V + V @ grad.conj().T @ Z)
		if not np.allclose(Z, project_unitary_tangent(V, Z)):
			G -= 0.5 * (Z @ V.conj().T + V @ Z.conj().T) @ grad
	dVlist[k] = G
	return np.stack(dVlist + dWlist)



def ansatz_hess_W(Vlist, Wlist, L, Z, k, cU, U, perms, unprojected=False):
	"""
		Assumes the Z_k grad. evaluation happens in the W gates layer.
		There are 4 possibilities for the i sum: i=k, i<k and i>k within 
		the W layer and i within the V layer.
	"""
	dVlist = [None for V in Vlist]
	dWlist = [None for W in Wlist]

	# i runs over Wlist, i>k.
	for i in range(k+1, len(Vlist)):
		U_tilde = np.eye(2**L).reshape([2]*2*L)
		for j in range(i+1, len(Wlist)):
			U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])
		U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(len(Vlist)):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
		U_tilde = (U @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(k):
			U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])
		U_tilde = ansatz_grad_directed_tensor(Wlist[k], U_tilde, L, Z, perms[k])
		for j in range(k+1, i):
			U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])

		dWi = ansatz_grad_tensor(Wlist[i], L, U_tilde, perms[i]).conj().T
		dWlist[i] = dWi if unprojected else  project_unitary_tangent(Wlist[i], dWi)

	# i runs over Wlist, k>i.
	for i in range(k):
		U_tilde = np.eye(2**L).reshape([2]*2*L)
		for j in range(i+1, k):
			U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])
		U_tilde = ansatz_grad_directed_tensor(Wlist[k], U_tilde, L, Z, perms[k])
		for j in range(k+1, len(Wlist)):
			U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])
		U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(len(Vlist)):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
		U_tilde = (U @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(i):
			U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])

		dWi = ansatz_grad_tensor(Wlist[i], L, U_tilde, perms[i]).conj().T
		dWlist[i] = dWi if unprojected else  project_unitary_tangent(Wlist[i], dWi)

	# i runs within Vlist.
	for i in range(len(Vlist)):
		U_tilde = np.eye(2**(L)).reshape([2]*2*L)
		for j in range(i+1, len(Vlist)):
			U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
		U_tilde = (U @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(k):
			U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])
		U_tilde = ansatz_grad_directed_tensor(Wlist[k], U_tilde, L, Z, perms[k])
		for j in range(k+1, len(Wlist)):
			U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])
		U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(i):
			U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])
		
		dVi = ansatz_grad_tensor(Vlist[i], L, U_tilde, perms[i]).conj().T
		dVlist[i] = dVi if unprojected else  project_unitary_tangent(Vlist[i], dVi)

	# i=k case.
	i = k
	U_tilde = np.eye(2**(L)).reshape([2]*2*L)
	for j in range(i+1, len(Wlist)):
		U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])
	U_tilde = (cU.conj().T @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
	for j in range(len(Vlist)):
		U_tilde = applyG_block_tensor(Vlist[j], U_tilde, L, perms[j])
	U_tilde = (U @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
	for j in range(i):
		U_tilde = applyG_block_tensor(Wlist[j], U_tilde, L, perms[j])
	G = ansatz_hess_single_layer_tensor(Wlist[k], L, Z, U_tilde, perms[k]).conj().T

	# Projection.
	if not unprojected:
		V = Wlist[k]
		G = project_unitary_tangent(V, G)
		grad = ansatz_grad_tensor(V, L, U_tilde, perms[k]).conj().T

		G -= 0.5 * (Z @ grad.conj().T @ V + V @ grad.conj().T @ Z)
		if not np.allclose(Z, project_unitary_tangent(V, Z)):
			G -= 0.5 * (Z @ V.conj().T + V @ Z.conj().T) @ grad
	dWlist[k] = G
	return np.stack(dVlist + dWlist)


def ansatz_hessian_matrix_tensor(Glist, cU, U, L, perms, flatten=True, unprojected=False):
	"""
	Construct the Hessian matrix.
	"""
	Vlist = Glist[:len(perms)]
	Wlist = Glist[len(perms):]
	eta = len(Vlist)

	Hess = np.zeros((2*eta, 16, 2*eta, 16))

	# k in V.
	for k in range(eta):
		for j in range(16):
			if unprojected:
				Z = np.zeros((4, 4), dtype=complex)
				Z.flat[j] = 1.0
			else:
				Z = np.zeros(16)
				Z[j] = 1
				Z = real_to_antisymm(np.reshape(Z, (4, 4)))

			dVZj = ansatz_hess_V(Vlist, Wlist, L, Vlist[k] @ Z, k, cU, U, perms, unprojected=unprojected)
			
			for i in range(2*eta):
				Hess[i, :, k, j] = dVZj[i].reshape(-1) if unprojected else \
						antisymm_to_real(antisymm( Glist[i].conj().T @ dVZj[i] )).reshape(-1)

	# k in W.
	for k in range(eta):
		for j in range(16):
			if unprojected:
				Z = np.zeros((4, 4), dtype=complex)
				Z.flat[j] = 1.0
			else:
				Z = np.zeros(16)
				Z[j] = 1
				Z = real_to_antisymm(np.reshape(Z, (4, 4)))

			dWZj = ansatz_hess_W(Vlist, Wlist, L, Wlist[k] @ Z, k, cU, U, perms, unprojected=unprojected)
			for i in range(2*eta):
				Hess[i, :, eta+k, j] = dWZj[i].reshape(-1) if unprojected else \
					antisymm_to_real(antisymm( Glist[i].conj().T @ dWZj[i] )).reshape(-1)
    
	if flatten:
		return Hess.reshape((2*eta*16, 2*eta*16))
	else:
		return Hess



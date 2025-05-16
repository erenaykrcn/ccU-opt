import numpy as np
from utils import (
	otimes, swap_matrix, permute_operation, applyG, applyG_block,
	partial_trace_keep, antisymm_to_real, antisymm, I2, X, Y, Z,
	project_unitary_tangent, real_to_antisymm
	)


def ansatz_hess_single_layer(V, L, Z, U_tilde, perm):
	"""
		Computes single permutation hessian for the i==k.
	"""
	G = np.zeros_like(V, dtype=complex)
	for i in range(L//2):
		k, l = (perm[2*i]+1, perm[2*i+1]+1)

		for z in range(L//2):
			if z==i:
				continue

			for j in range(i):
				k_, l_ = (perm[2*j]+1, perm[2*j+1]+1)
				if z==j:
					U_tilde = applyG(Z.conj(), k_, l_, L+1) @ U_tilde
				else:
					U_tilde = applyG(V.conj(), k_, l_, L+1) @ U_tilde

			for j in list(range(i+1, L//2))[::-1]:
				k_, l_ = (perm[2*j]+1, perm[2*j+1]+1)
				if z==j:
					U_tilde = U_tilde @ applyG(Z.conj(), k_, l_, L+1)
				else:
					U_tilde = U_tilde @ applyG(V.conj(), k_, l_, L+1)

			# Take partial trace wrt all qubits but 0, k, l.
			T = partial_trace_keep(U_tilde, [0, k, l], L+1)
			G += T

	# Add the additional terms resulting from unitary projection.
	G = project_unitary_tangent(V, G)
	grad = ansatz_grad(V, L, U_tilde, perm)
	G -= 0.5 * (Z @ grad.conj().T @ V + V @ grad.conj().T @ Z)
	if not np.allclose(Z, project_unitary_tangent(V, Z)):
		G -= 0.5 * (Z @ V.conj().T + V @ Z.conj().T) @ grad
	return G


def ansatz_grad_directed(V, L, Z, perm):
	assert V.shape == (8, 8)
	assert Z.shape == (8, 8)
	assert L % 2 == 0

	G = np.zeros((2**(L+1), 2**(L+1)), dtype=complex)
	for i in range(L//2):
		U = np.eye(2**(L+1))
		_, k, l = (0, perm[2*i]+1, perm[2*i+1]+1)

		for j in range(i):
			k_, l_ = (perm[2*j]+1, perm[2*j+1]+1)
			U = applyG(V, k_, l_, L+1) @ U

		U = applyG(Z, k, l, L+1) @ U

		for j in list(range(i+1, L//2))[::-1]:
			k_, l_ = (perm[2*j]+1, perm[2*j+1]+1)
			U = applyG(V, k_, l_, L+1) @ U

		G += U
	return G


def ansatz_hess_V(Vlist, Wlist, L, Z, k, cU, U, perms):
	"""
		Assumes the Z_k grad. evaluation happens in the V gates layer.
		There are 4 possibilities for the i sum: i=k, i<k and i>k within 
		the V layer and i within the W layer. 
	"""
	dVlist = [None for V in Vlist]
	dWlist = [None for W in Wlist]

	# i runs over Vlist, i>k
	for i in range(k+1, len(Vlist)):
		U_tilde = np.eye(2**(L+1))
		for j in range(i+1, len(Vlist)):
			U_tilde = applyG_block(Vlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = otimes([I2, U]).conj().T @ U_tilde
		for j in range(len(Wlist)):
			U_tilde = applyG_block(Wlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = cU @ U_tilde
		for j in range(k):
			U_tilde = applyG_block(Vlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = ansatz_grad_directed(Vlist[k], L, Z, perms[k]).conj().T @ U_tilde
		for j in range(k+1, i):
			U_tilde = applyG_block(Vlist[j], L, perms[j]).conj().T @ U_tilde

		dVi = ansatz_grad(Vlist[i], L, U_tilde, perms[i])
		dVlist[i] = project_unitary_tangent(Vlist[i], dVi)

	# i runs over Vlist, k>i
	for i in range(k):
		U_tilde = np.eye(2**(L+1))
		for j in range(i+1, k):
			U_tilde = applyG_block(Vlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = ansatz_grad_directed(Vlist[k], L, Z, perms[k]).conj().T @ U_tilde
		for j in range(k+1, len(Vlist)):
			U_tilde = applyG_block(Vlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = otimes([I2, U]).conj().T @ U_tilde
		for j in range(len(Wlist)):
			U_tilde = applyG_block(Wlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = cU @ U_tilde
		for j in range(i):
			U_tilde = applyG_block(Vlist[j], L, perms[j]).conj().T @ U_tilde

		dVi = ansatz_grad(Vlist[i], L, U_tilde, perms[i])
		dVlist[i] = project_unitary_tangent(Vlist[i], dVi)

	# i runs within Wlist.
	for i in range(len(Wlist)):
		U_tilde = np.eye(2**(L+1))
		for j in range(i+1, len(Wlist)):
			U_tilde = applyG_block(Wlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = cU @ U_tilde
		for j in range(k):
			U_tilde = applyG_block(Vlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = ansatz_grad_directed(Vlist[k], L, Z, perms[k]).conj().T @ U_tilde
		for j in range(k+1, len(Vlist)):
			U_tilde = applyG_block(Vlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = otimes([I2, U]).conj().T @ U_tilde
		for j in range(i):
			U_tilde = applyG_block(Wlist[j], L, perms[j]).conj().T @ U_tilde
		
		dWi = ansatz_grad(Wlist[i], L, U_tilde, perms[i])
		dWlist[i] = project_unitary_tangent(Wlist[i], dWi)

	# i=k case.
	i = k
	U_tilde = np.eye(2**(L+1))
	for j in range(i+1, len(Vlist)):
		U_tilde = applyG_block(Vlist[j], L, perms[j]).conj().T @ U_tilde
	U_tilde = otimes([I2, U]).conj().T @ U_tilde
	for j in range(len(Wlist)):
		U_tilde = applyG_block(Wlist[j], L, perms[j]).conj().T @ U_tilde
	U_tilde = cU @ U_tilde
	for j in range(i):
		U_tilde = applyG_block(Vlist[j], L, perms[j]).conj().T @ U_tilde
	dVlist[k] = ansatz_hess_single_layer(Vlist[k], L, Z, U_tilde, perms[k])

	return np.stack(dVlist + dWlist)



def ansatz_hess_W(Vlist, Wlist, L, Z, k, cU, U, perms):
	"""
		Assumes the Z_k grad. evaluation happens in the W gates layer.
		There are 4 possibilities for the i sum: i=k, i<k and i>k within 
		the W layer and i within the V layer.
	"""
	dVlist = [None for V in Vlist]
	dWlist = [None for W in Wlist]

	# i runs over Wlist, i>k.
	for i in range(k+1, len(Vlist)):
		U_tilde = np.eye(2**(L+1))
		for j in range(i+1, len(Wlist)):
			U_tilde = applyG_block(Wlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = cU @ U_tilde
		for j in range(len(Vlist)):
			U_tilde = applyG_block(Vlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = otimes([I2, U]).conj().T @ U_tilde
		for j in range(k):
			U_tilde = applyG_block(Wlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = ansatz_grad_directed(Wlist[k], L, Z, perms[k]).conj().T @ U_tilde
		for j in range(k+1, i):
			U_tilde = applyG_block(Wlist[j], L, perms[j]).conj().T @ U_tilde

		dWi = ansatz_grad(Wlist[i], L, U_tilde, perms[i])
		dWlist[i] = project_unitary_tangent(Wlist[i], dWi)

	# i runs over Wlist, k>i.
	for i in range(k):
		U_tilde = np.eye(2**(L+1))
		for j in range(i+1, k):
			U_tilde = applyG_block(Wlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = ansatz_grad_directed(Wlist[k], L, Z, perms[k]).conj().T @ U_tilde
		for j in range(k+1, len(Wlist)):
			U_tilde = applyG_block(Wlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = cU @ U_tilde
		for j in range(len(Vlist)):
			U_tilde = applyG_block(Vlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = otimes([I2, U]).conj().T @ U_tilde
		for j in range(i):
			U_tilde = applyG_block(Wlist[j], L, perms[j]).conj().T @ U_tilde

		dWi = ansatz_grad(Wlist[i], L, U_tilde, perms[i])
		dWlist[i] = project_unitary_tangent(Wlist[i], dWi)

	# i runs within Vlist.
	for i in range(len(Vlist)):
		U_tilde = np.eye(2**(L+1))
		for j in range(i+1, len(Vlist)):
			U_tilde = applyG_block(Vlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = otimes([I2, U]).conj().T @ U_tilde
		for j in range(k):
			U_tilde = applyG_block(Wlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = ansatz_grad_directed(Wlist[k], L, Z, perms[k]).conj().T @ U_tilde
		for j in range(k+1, len(Wlist)):
			U_tilde = applyG_block(Wlist[j], L, perms[j]).conj().T @ U_tilde
		U_tilde = cU @ U_tilde
		for j in range(i):
			U_tilde = applyG_block(Vlist[j], L, perms[j]).conj().T @ U_tilde
		
		dVi = ansatz_grad(Vlist[i], L, U_tilde, perms[i])
		dVlist[i] = project_unitary_tangent(Vlist[i], dVi)

	# i=k case.
	i = k
	U_tilde = np.eye(2**(L+1))
	for j in range(i+1, len(Wlist)):
		U_tilde = applyG_block(Wlist[j], L, perms[j]).conj().T @ U_tilde
	U_tilde = cU @ U_tilde
	for j in range(len(Vlist)):
		U_tilde = applyG_block(Vlist[j], L, perms[j]).conj().T @ U_tilde
	U_tilde = otimes([I2, U]).conj().T @ U_tilde
	for j in range(i):
		U_tilde = applyG_block(Wlist[j], L, perms[j]).conj().T @ U_tilde
	dWlist[k] = ansatz_hess_single_layer(Wlist[k], L, Z, U_tilde, perms[k])

	return np.stack(dVlist + dWlist)


def ansatz_hessian_matrix(Glist, cU, U, L, perms):
	"""
	Construct the Hessian matrix.
	"""
	Vlist = Glist[:len(perms)]
	Wlist = Glist[len(perms):]
	eta = len(Vlist)

	Hess = np.zeros((2*eta, 64, 2*eta, 64))

	# k in V.
	for k in range(eta):
		for j in range(64):
			# unit vector
			Z = np.zeros(64)
			Z[j] = 1
			Z = real_to_antisymm(np.reshape(Z, (8, 8)))
			dVZj = ansatz_hess_V(Vlist, Wlist, L, Vlist[k] @ Z, k, cU, U, perms)
			for i in range(2*eta):
				Hess[i, :, k, j] = antisymm_to_real(antisymm( Glist[i].conj().T @ dVZj[i] )).reshape(-1)

	# k in W.
	for k in range(eta):
		for j in range(64):
			# unit vector
			Z = np.zeros(64)
			Z[j] = 1
			Z = real_to_antisymm(np.reshape(Z, (8, 8)))
			dWZj = ansatz_hess_W(Vlist, Wlist, L, Wlist[k] @ Z, k, cU, U, perms)
			for i in range(2*eta):
				Hess[i, :, eta+k, j] = antisymm_to_real(antisymm( Glist[i].conj().T @ dWZj[i] )).reshape(-1)
    
	return Hess.reshape((2*eta*64, 2*eta*64))



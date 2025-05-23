import numpy as np
from ansatz import ansatz_grad
from utils import (
	otimes, swap_matrix, permute_operation, applyG, applyG_block,
	partial_trace_keep, antisymm_to_real, antisymm, I2, X, Y, Z,
	project_unitary_tangent, real_to_antisymm
	)


def ansatz_hess_single_layer(V, L, Z, U_tilde_, perm):
	"""
		Computes single permutation hessian for the i==k.
	"""
	G = np.zeros_like(V, dtype=complex)
	for i in range(L//2):
		U_tilde = U_tilde_
		k, l = (perm[2*i], perm[2*i+1])

		for z in range(L//2):
			if z==i:
				continue

			for j in range(i):
				k_, l_ = (perm[2*j], perm[2*j+1])
				if z==j:
					U_tilde = applyG(Z, k_, l_, L) @ U_tilde
				else:
					U_tilde = applyG(V, k_, l_, L) @ U_tilde

			for j in list(range(i+1, L//2))[::-1]:
				k_, l_ = (perm[2*j], perm[2*j+1])
				if z==j:
					U_tilde = U_tilde @ applyG(Z, k_, l_, L)
				else:
					U_tilde = U_tilde @ applyG(V, k_, l_, L)

			# Take partial trace wrt all qubits but k, l.
			T = partial_trace_keep(U_tilde, [k, l], L)
			if k>l:
				# Partial trace interprets the qubit l as the first qubit of 
				# the resulting two qubit gate. We need to fix that.
				SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
				T = SWAP @ T @ SWAP
			G += T
	return G


def ansatz_grad_directed(V, L, Z, perm):
	assert V.shape == (4, 4)
	assert Z.shape == (4, 4)
	assert L % 2 == 0

	G = np.zeros((2**L, 2**L), dtype=complex)
	for i in range(L//2):
		U = np.eye(2**L)
		k, l = (perm[2*i], perm[2*i+1])

		for j in range(i):
			k_, l_ = (perm[2*j], perm[2*j+1])
			U = applyG(V, k_, l_, L) @ U

		U = applyG(Z, k, l, L) @ U

		for j in list(range(i+1, L//2))[::-1]:
			k_, l_ = (perm[2*j], perm[2*j+1])
			U = applyG(V, k_, l_, L) @ U

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
		U_tilde = np.eye(2**L)
		for j in range(i+1, len(Vlist)):
			U_tilde = applyG_block(Vlist[j], L, perms[j]) @ U_tilde
		U_tilde = U @ U_tilde
		for j in range(len(Wlist)):
			U_tilde = applyG_block(Wlist[j], L, perms[j]) @ U_tilde
		U_tilde = cU.conj().T @ U_tilde
		for j in range(k):
			U_tilde = applyG_block(Vlist[j], L, perms[j]) @ U_tilde
		U_tilde = ansatz_grad_directed(Vlist[k], L, Z, perms[k]) @ U_tilde
		for j in range(k+1, i):
			U_tilde = applyG_block(Vlist[j], L, perms[j]) @ U_tilde

		dVi = ansatz_grad(Vlist[i], L, U_tilde, perms[i]).conj().T
		dVlist[i] = project_unitary_tangent(Vlist[i], dVi) # TODO: Perhaps move the conj().T here from up?

	# i runs over Vlist, k>i
	for i in range(k):
		U_tilde = np.eye(2**L)
		for j in range(i+1, k):
			U_tilde = applyG_block(Vlist[j], L, perms[j]) @ U_tilde
		U_tilde = ansatz_grad_directed(Vlist[k], L, Z, perms[k]) @ U_tilde
		for j in range(k+1, len(Vlist)):
			U_tilde = applyG_block(Vlist[j], L, perms[j]) @ U_tilde
		U_tilde = U @ U_tilde
		for j in range(len(Wlist)):
			U_tilde = applyG_block(Wlist[j], L, perms[j]) @ U_tilde
		U_tilde = cU.conj().T @ U_tilde
		for j in range(i):
			U_tilde = applyG_block(Vlist[j], L, perms[j]) @ U_tilde

		dVi = ansatz_grad(Vlist[i], L, U_tilde, perms[i]).conj().T
		dVlist[i] = project_unitary_tangent(Vlist[i], dVi)

	# i runs within Wlist.
	for i in range(len(Wlist)):
		U_tilde = np.eye(2**(L))
		for j in range(i+1, len(Wlist)):
			U_tilde = applyG_block(Wlist[j], L, perms[j]) @ U_tilde
		U_tilde = cU.conj().T @ U_tilde
		for j in range(k):
			U_tilde = applyG_block(Vlist[j], L, perms[j]) @ U_tilde
		U_tilde = ansatz_grad_directed(Vlist[k], L, Z, perms[k]) @ U_tilde
		for j in range(k+1, len(Vlist)):
			U_tilde = applyG_block(Vlist[j], L, perms[j]) @ U_tilde
		U_tilde = U @ U_tilde
		for j in range(i):
			U_tilde = applyG_block(Wlist[j], L, perms[j]) @ U_tilde
		
		dWi = ansatz_grad(Wlist[i], L, U_tilde, perms[i]).conj().T
		dWlist[i] = project_unitary_tangent(Wlist[i], dWi)

	# i=k case.
	i = k
	U_tilde = np.eye(2**L)
	for j in range(i+1, len(Vlist)):
		U_tilde = applyG_block(Vlist[j], L, perms[j]) @ U_tilde
	U_tilde = U @ U_tilde
	for j in range(len(Wlist)):
		U_tilde = applyG_block(Wlist[j], L, perms[j]) @ U_tilde
	U_tilde = cU.conj().T @ U_tilde
	for j in range(i):
		U_tilde = applyG_block(Vlist[j], L, perms[j]) @ U_tilde

	G = ansatz_hess_single_layer(Vlist[k], L, Z, U_tilde, perms[k]).conj().T

	# Projection.
	V = Vlist[k]
	G = project_unitary_tangent(V, G)
	grad = ansatz_grad(V, L, U_tilde, perms[k]).conj().T # TODO: Check this transpose.
	G -= 0.5 * (Z @ grad.conj().T @ V + V @ grad.conj().T @ Z)
	if not np.allclose(Z, project_unitary_tangent(V, Z)):
		G -= 0.5 * (Z @ V.conj().T + V @ Z.conj().T) @ grad
	dVlist[k] = G
	
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
		U_tilde = np.eye(2**L)
		for j in range(i+1, len(Wlist)):
			U_tilde = applyG_block(Wlist[j], L, perms[j]) @ U_tilde
		U_tilde = cU.conj().T @ U_tilde
		for j in range(len(Vlist)):
			U_tilde = applyG_block(Vlist[j], L, perms[j]) @ U_tilde
		U_tilde = U @ U_tilde
		for j in range(k):
			U_tilde = applyG_block(Wlist[j], L, perms[j]) @ U_tilde
		U_tilde = ansatz_grad_directed(Wlist[k], L, Z, perms[k]) @ U_tilde
		for j in range(k+1, i):
			U_tilde = applyG_block(Wlist[j], L, perms[j]) @ U_tilde

		dWi = ansatz_grad(Wlist[i], L, U_tilde, perms[i]).conj().T
		dWlist[i] = project_unitary_tangent(Wlist[i], dWi)

	# i runs over Wlist, k>i.
	for i in range(k):
		U_tilde = np.eye(2**L)
		for j in range(i+1, k):
			U_tilde = applyG_block(Wlist[j], L, perms[j]) @ U_tilde
		U_tilde = ansatz_grad_directed(Wlist[k], L, Z, perms[k]) @ U_tilde
		for j in range(k+1, len(Wlist)):
			U_tilde = applyG_block(Wlist[j], L, perms[j]) @ U_tilde
		U_tilde = cU.conj().T @ U_tilde
		for j in range(len(Vlist)):
			U_tilde = applyG_block(Vlist[j], L, perms[j]) @ U_tilde
		U_tilde = U @ U_tilde
		for j in range(i):
			U_tilde = applyG_block(Wlist[j], L, perms[j]) @ U_tilde

		dWi = ansatz_grad(Wlist[i], L, U_tilde, perms[i]).conj().T
		dWlist[i] = project_unitary_tangent(Wlist[i], dWi)

	# i runs within Vlist.
	for i in range(len(Vlist)):
		U_tilde = np.eye(2**(L))
		for j in range(i+1, len(Vlist)):
			U_tilde = applyG_block(Vlist[j], L, perms[j]) @ U_tilde
		U_tilde = U @ U_tilde
		for j in range(k):
			U_tilde = applyG_block(Wlist[j], L, perms[j]) @ U_tilde
		U_tilde = ansatz_grad_directed(Wlist[k], L, Z, perms[k]) @ U_tilde
		for j in range(k+1, len(Wlist)):
			U_tilde = applyG_block(Wlist[j], L, perms[j]) @ U_tilde
		U_tilde = cU.conj().T @ U_tilde
		for j in range(i):
			U_tilde = applyG_block(Vlist[j], L, perms[j]) @ U_tilde
		
		dVi = ansatz_grad(Vlist[i], L, U_tilde, perms[i]).conj().T
		dVlist[i] = project_unitary_tangent(Vlist[i], dVi)

	# i=k case.

	i = k
	U_tilde = np.eye(2**(L))
	for j in range(i+1, len(Wlist)):
		U_tilde = applyG_block(Wlist[j], L, perms[j]) @ U_tilde
	U_tilde = cU.conj().T @ U_tilde
	for j in range(len(Vlist)):
		U_tilde = applyG_block(Vlist[j], L, perms[j]) @ U_tilde
	U_tilde = U @ U_tilde
	for j in range(i):
		U_tilde = applyG_block(Wlist[j], L, perms[j]) @ U_tilde
	G = ansatz_hess_single_layer(Wlist[k], L, Z, U_tilde, perms[k]).conj().T

	# Projection.
	V = Wlist[k]
	G = project_unitary_tangent(V, G)
	grad = ansatz_grad(V, L, U_tilde, perms[k]).conj().T # TODO: Check this transpose.
	G -= 0.5 * (Z @ grad.conj().T @ V + V @ grad.conj().T @ Z)
	if not np.allclose(Z, project_unitary_tangent(V, Z)):
		G -= 0.5 * (Z @ V.conj().T + V @ Z.conj().T) @ grad
	dWlist[k] = G

	return np.stack(dVlist + dWlist)


def ansatz_hessian_matrix(Glist, cU, U, L, perms):
	"""
	Construct the Hessian matrix.

	Major TODOs:
	-> Modify the conj().T such that only cU is transposed, none of the
	V,Ws are transposed but then at the end we transpose the individual
	df/dG elements (after the sum, before the stacking).

	-> Have to check if the case k>l causes additional problems within hessian functions.
	"""
	Vlist = Glist[:len(perms)]
	Wlist = Glist[len(perms):]
	eta = len(Vlist)

	Hess = np.zeros((2*eta, 16, 2*eta, 16))

	# k in V.
	for k in range(eta):
		for j in range(16):
			# unit vector
			Z = np.zeros(16)
			Z[j] = 1
			Z = real_to_antisymm(np.reshape(Z, (4, 4)))

			dVZj = ansatz_hess_V(Vlist, Wlist, L, Vlist[k] @ Z, k, cU, U, perms)
			
			for i in range(2*eta):
				Hess[i, :, k, j] = antisymm_to_real(antisymm( Glist[i].conj().T @ dVZj[i] )).reshape(-1)

	# k in W.
	for k in range(eta):
		for j in range(16):
			# unit vector
			Z = np.zeros(16)
			Z[j] = 1
			Z = real_to_antisymm(np.reshape(Z, (4, 4)))
			dWZj = ansatz_hess_W(Vlist, Wlist, L, Wlist[k] @ Z, k, cU, U, perms)
			for i in range(2*eta):
				Hess[i, :, eta+k, j] = antisymm_to_real(antisymm( Glist[i].conj().T @ dWZj[i] )).reshape(-1)
    
	return Hess.reshape((2*eta*16, 2*eta*16))



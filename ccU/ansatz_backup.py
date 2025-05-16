import numpy as np
from utils import (
	otimes, swap_matrix, permute_operation, applyG, applyG_block,
	partial_trace_keep, antisymm_to_real, antisymm, I2, X, Y, Z,
	project_unitary_tangent, real_to_antisymm
	)


def ansatz(Glist, U, L, perms):
	"""
		Our circuit Ansatz will wrap the time evolution operator U
		through couplings of neighbour interactions and the ancilla 
		qubit. For each permutation of neighbours, there is another 
		three qubit gate V in Vlist. 
	"""
	assert len(Glist) == 2*len(perms)
	assert Glist[0].shape == (8, 8) # 3 qubit gate
	Vs = Glist[:len(perms)]
	Ws = Glist[len(perms):] 
	cU = np.eye(2**(L+1))

	for i, V in enumerate(Vs):
		cU = applyG_block(V, L, perms[i]) @ cU
	cU = np.kron(I2, U) @ cU
	for i, W in enumerate(Ws):
		cU = applyG_block(W, L, perms[i]) @ cU

	return cU


def ansatz_grad(V, L, U_tilde, perm):
	assert V.shape == (8, 8)
	assert U_tilde.shape == (2**(L+1), 2**(L+1))
	assert L % 2 == 0

	G = np.zeros_like(V, dtype=complex)
	for i in range(L//2):
		_, k, l = (0, perm[2*i]+1, perm[2*i+1]+1)

    	# Contract U_tilde and V gates for all perm qubits but k,l,m.
    	# Careful about adding it before or after.

    	# TODO: Is V.conj() correcT???
		for j in range(i):
			k_, l_ = (perm[2*j]+1, perm[2*j+1]+1)
			U_tilde = applyG(V.conj(), k_, l_, L+1) @ U_tilde

		for j in list(range(i+1, L//2))[::-1]:
			k_, l_ = (perm[2*j]+1, perm[2*j+1]+1)
			U_tilde = U_tilde @ applyG(V.conj(), k_, l_, L+1)

		# Take partial trace wrt all qubits but 0, k, l.
		T = partial_trace_keep(U_tilde, [0, k, l], L+1)

		G += T
	return G


def ansatz_grad_vector(Glist, cU, U, L, perms, flatten=True):
	Vlist = Glist[:len(perms)]
	Wlist = Glist[len(perms):]
	grads_V = []
	for i, V in enumerate(Vlist):
		U_tilde = cU
		for j in range(i):
			U_tilde = applyG_block(Vlist[j], L, perms[j]).conj().T @ U_tilde
		for j in range(len(perms)-1, -1, -1):
			U_tilde = U_tilde @ applyG_block(Wlist[j], L, perms[j]).conj().T
		U_tilde = U_tilde @ otimes([I2, U]).conj().T
		for j in range(len(perms)-1, i, -1):
			U_tilde = U_tilde @ applyG_block(Vlist[j], L, perms[j]).conj().T
		grads_V.append(ansatz_grad(V, L, U_tilde, perms[i]))

	grads_W = []
	for i, W in enumerate(Wlist):
		U_tilde = otimes([I2, U]).conj().T
		for j in range(i):
			U_tilde = applyG_block(Wlist[j], L, perms[j]).conj().T @ U_tilde
		for j in range(len(perms)-1, -1, -1):
			U_tilde = U_tilde @ applyG_block(Vlist[j], L, perms[j]).conj().T
		U_tilde = U_tilde @ cU
		for j in range(len(perms)-1, i, -1):
			U_tilde = U_tilde @ applyG_block(Wlist[j], L, perms[j]).conj().T
		grads_W.append(ansatz_grad(W, L, U_tilde, perms[i]))


	grad = np.stack(grads_V + grads_W)

	# Project onto tangent space.
	if flatten:
		return np.stack([antisymm_to_real(
	        antisymm(Glist[j].conj().T @ grad[j]))
	        for j in range(len(grad))])\
		.reshape(-1)
	else:
		return np.stack([antisymm_to_real(
	        antisymm(Glist[j].conj().T @ grad[j]))
	        for j in range(len(grad))])


















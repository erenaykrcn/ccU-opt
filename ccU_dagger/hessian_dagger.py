import numpy as np
from ansatz_dagger import ansatz_grad_dagger, ansatz_grad_vector_dagger
from utils_dagger import (
	applyG_tensor, applyG_block_tensor,
	partial_trace_keep, antisymm_to_real, antisymm,
	project_unitary_tangent, real_to_antisymm
	)
from hessian_utils import ansatz_grad_directed_dagger1, ansatz_grad_directed_dagger2


def ansatz_hess_single_layer_dagger(A, B, V, L, Z, perm):
	"""
		Computes single permutation hessian for the i==k.
	"""
	G = np.zeros_like(V, dtype=complex)

	for layer in range(2):
		for i in range(L//2):
			k, l = (perm[2*i], perm[2*i+1])
			for z in range(L//2):
				U_tilde = (A.reshape(2**L, 2**L) @ applyG_block_tensor(
					V.conj().T, B, L, perm).reshape(2**L, 2**L)).reshape([2]*2*L
				) if layer==0 else (B.reshape(2**L, 2**L) @ applyG_block_tensor(
					V, A, L, perm).reshape(2**L, 2**L) ).reshape([2]*2*L)

				if z==i:
					continue
				for j in range(i):
					k_, l_ = (perm[2*j], perm[2*j+1])
					if z==j:
						U_tilde = applyG_tensor(Z if layer==0 else Z.conj().T, U_tilde, k_, l_)
					else:
						U_tilde = applyG_tensor(V if layer==0 else V.conj().T, U_tilde, k_, l_)
				for j in range(i+1, L//2):
					k_, l_ = (perm[2*j], perm[2*j+1])
					if z==j:
						U_tilde = applyG_tensor(Z if layer==0 else Z.conj().T, U_tilde, k_, l_) 
					else:
						U_tilde = applyG_tensor(V if layer==0 else V.conj().T, U_tilde, k_, l_)

				T = partial_trace_keep(U_tilde.reshape(2**L, 2**L), [k, l], L)
				if k>l:
					SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
					T = SWAP @ T @ SWAP
				G += T if layer==0 else T.conj().T

			for z in range(L//2):
				U_tilde = B if layer==0 else A
				for j in range(L//2):
					k_, l_ = (perm[2*j], perm[2*j+1])
					if z==j:
						U_tilde = applyG_tensor(Z.conj().T if layer==0 else Z, U_tilde, k_, l_)
					else:
						U_tilde = applyG_tensor(V.conj().T if layer==0 else V, U_tilde, k_, l_)
				U_tilde = ((A if layer==0 else B).reshape(2**L, 2**L) @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
				for j in range(i):
					k_, l_ = (perm[2*j], perm[2*j+1])
					U_tilde = applyG_tensor(V if layer==0 else V.conj().T, U_tilde, k_, l_)
				for j in range(i+1, L//2):
					k_, l_ = (perm[2*j], perm[2*j+1])
					U_tilde = applyG_tensor(V if layer==0 else V.conj().T, U_tilde, k_, l_)
					
				T = partial_trace_keep(U_tilde.reshape(2**L, 2**L), [k, l], L)
				if k>l:
					SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
					T = SWAP @ T @ SWAP
				G += T if layer==0 else T.conj().T
	return G


def ansatz_hess(Glist, L, Z, k, cU, U, perms, unprojected=False):
	"""
		Assumes the Z_k grad. evaluation happens in the V gates layer.
		There are 4 possibilities for the i sum: i=k, i<k and i>k within 
		the V layer and i within the W layer. 
	"""
	eta = len(perms)
	dGlist = [None for G in Glist]

	# i>k
	for i in range(k+1, eta):
		B = np.eye(2**L).reshape([2]*2*L)
		for j in range(i+1, eta):
			B = applyG_block_tensor(Glist[j], B, L, perms[j])
		B = (cU.conj().T @ B.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(eta-1, i, -1):
			B = applyG_block_tensor(Glist[j].conj().T, B, L, perms[j])
		
		A1 = np.eye(2**L).reshape([2]*2*L)
		for j in range(i-1, k, -1):
			A1 = applyG_block_tensor(Glist[j].conj().T, A1, L, perms[j])
		
		#A = ansatz_grad_directed_dagger(Glist[k].conj().T, A, L, Z.conj().T, perms[k], conj=True)
		
		A2 = np.eye(2**L).reshape([2]*2*L)
		for j in range(k-1, -1, -1):
			A2 = applyG_block_tensor(Glist[j].conj().T, A2, L, perms[j])
		A2 = (U @ A2.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(k):
			A2 = applyG_block_tensor(Glist[j], A2, L, perms[j])
		
		#A = ansatz_grad_directed_dagger(Glist[k], A, L, Z, perms[k])
		
		A3 = np.eye(2**L).reshape([2]*2*L)
		for j in range(k+1, i):
			A3 = applyG_block_tensor(Glist[j], A3, L, perms[j])

		dGi = ansatz_grad_directed_dagger1(B, A1, A2, A3, Glist, Z, i, k, L, perms).conj().T
		dGlist[i] = dGi if unprojected else project_unitary_tangent(Glist[i], dGi)
		
	# k>i
	for i in range(k):
		B1 = np.eye(2**L).reshape([2]*2*L)
		for j in range(i+1, k):
			B1 = applyG_block_tensor(Glist[j], B1, L, perms[j])
		
		#B = ansatz_grad_directed_dagger(Glist[k], B, L, Z, perms[k])
		
		B2 = np.eye(2**L).reshape([2]*2*L)
		for j in range(k+1, eta):
			B2 = applyG_block_tensor(Glist[j], B2, L, perms[j])
		B2 = (cU.conj().T @ B2.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(eta-1, k, -1):
			B2 = applyG_block_tensor(Glist[j].conj().T, B2, L, perms[j])
		
		#B = ansatz_grad_directed_dagger(Glist[k].conj(), B, L, Z.conj().T, perms[k], conj=True) #??
		
		B3 = np.eye(2**L).reshape([2]*2*L)
		for j in range(k-1, i, -1):
			B3 = applyG_block_tensor(Glist[j].conj().T, B3, L, perms[j])

		A = np.eye(2**L).reshape([2]*2*L)
		for j in range(i-1, -1, -1):
			A = applyG_block_tensor(Glist[j].conj().T, A, L, perms[j])
		A = (U @ A.reshape(2**L, 2**L)).reshape([2]*2*L)
		for j in range(i):
			A = applyG_block_tensor(Glist[j], A, L, perms[j])

		#dGi = ansatz_grad_dagger(A, B, Glist[i], L, perms[i]).conj().T
		dGi = ansatz_grad_directed_dagger2(B1, B2, B3, A, Glist, Z, i, k, L, perms).conj().T
		dGlist[i] = dGi if unprojected else project_unitary_tangent(Glist[i], dGi)
		
	# i=k case.
	i = k
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

	G = ansatz_hess_single_layer_dagger(A, B, Glist[k], L, Z, perms[k]).conj().T

	# Projection.
	if not unprojected:
		V = Glist[k]
		G = project_unitary_tangent(V, G)
		grad = ansatz_grad_dagger(A, B, V, L, perms[k]).conj().T

		G -= 0.5 * (Z @ grad.conj().T @ V + V @ grad.conj().T @ Z)
		if not np.allclose(Z, project_unitary_tangent(V, Z)):
			G -= 0.5 * (Z @ V.conj().T + V @ Z.conj().T) @ grad
	dGlist[k] = G
	return dGlist



def ansatz_hessian_matrix_dagger(Glist, cU, U, L, perms, flatten=True, unprojected=False):
	"""
	Construct the Hessian matrix.
	"""
	eta = len(Glist)
	Hess = np.zeros((eta, 16, eta, 16))

	for k in range(eta):
		for j in range(16):
			Z = np.zeros(16)
			Z[j] = 1
			Z = real_to_antisymm(np.reshape(Z, (4, 4)))

			dGZj = ansatz_hess(Glist, L, Glist[k] @ Z, k, cU, U, perms, unprojected=unprojected)
			
			for i in range(eta):
				Hess[i, :, k, j] = dGZj[i].reshape(-1) if unprojected else \
						antisymm_to_real(antisymm( Glist[i].conj().T @ dGZj[i] )).reshape(-1)
	if flatten:
		return Hess.reshape((eta*16, eta*16))
	else:
		return Hess



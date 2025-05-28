import numpy as np
from ansatz_dagger import ansatz_grad_dagger, ansatz_grad_vector_dagger
from utils_dagger import (
	applyG_tensor, applyG_block_tensor,
	partial_trace_keep, antisymm_to_real, antisymm,
	project_unitary_tangent, real_to_antisymm
	)


def ansatz_grad_directed_dagger1(B, A1, A2, A3, Glist, Z, i, k, L, perms):
	# k < i
	G = np.zeros_like(Glist[i], dtype=complex)
	for layer in range(2):
		for z in range(L//2):
			# z in Z
			if layer==0:
				U_tilde = (A1.reshape(2**L, 2**L) @ applyG_block_tensor(Glist[i].conj().T, B, L, 
					perms[i]).reshape(2**L, 2**L)).reshape([2]*2*L)
				U_tilde = (A2.reshape(2**L, 2**L) @ applyG_block_tensor(Glist[k].conj().T, U_tilde, L, 
					perms[k]).reshape(2**L, 2**L)).reshape([2]*2*L)
				for j in range(L//2):
					k_, l_ = (perms[k][2*j], perms[k][2*j+1])
					if z==j:
						U_tilde = applyG_tensor(Z, U_tilde, k_, l_)
					else:
						U_tilde = applyG_tensor(Glist[k], U_tilde, k_, l_)
				U_tilde = (A3.reshape(2**L, 2**L) @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
			else:
				U_tilde = (A2.reshape(2**L, 2**L) @ applyG_block_tensor(Glist[k].conj().T, A1, L, 
					perms[k]).reshape(2**L, 2**L)).reshape([2]*2*L)
				for j in range(L//2):
					k_, l_ = (perms[k][2*j], perms[k][2*j+1])
					if z==j:
						U_tilde = applyG_tensor(Z, U_tilde, k_, l_)
					else:
						U_tilde = applyG_tensor(Glist[k], U_tilde, k_, l_)
				U_tilde = (A3.reshape(2**L, 2**L) @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
				U_tilde = (B.reshape(2**L, 2**L) @ applyG_block_tensor(Glist[i], U_tilde, L, 
					perms[i]).reshape(2**L, 2**L)).reshape([2]*2*L)


			for g in range(L//2):
				k__, l__ = (perms[i][2*g], perms[i][2*g+1])
				U_working = U_tilde.copy()

				for j in range(L//2):
					if j==g:
						continue
					else:
						k_, l_ = perms[i][2 * j], perms[i][2 * j + 1]
						U_working = applyG_tensor(Glist[i] if layer==0 else Glist[i].conj().T, 
							U_working, k_, l_)

				T = partial_trace_keep(U_working.reshape(2**L, 2**L), [k__, l__], L)
				if k__ > l__:
					SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
					T = SWAP @ T @ SWAP
				G += T if layer==0 else T.conj().T


		for z in range(L//2):
			# z in Z^{\dag}
			if layer==0:
				U_tilde = (A1.reshape(2**L, 2**L) @ applyG_block_tensor(Glist[i].conj().T, B, L, 
					perms[i]).reshape(2**L, 2**L)).reshape([2]*2*L)
				for j in range(L//2):
					k_, l_ = (perms[k][2*j], perms[k][2*j+1])
					if z==j:
						U_tilde = applyG_tensor(Z.conj().T, U_tilde, k_, l_)
					else:
						U_tilde = applyG_tensor(Glist[k].conj().T, U_tilde, k_, l_)
				U_tilde = (A2.reshape(2**L, 2**L) @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
				U_tilde = applyG_block_tensor(Glist[k], U_tilde, L, perms[k])
				U_tilde = (A3.reshape(2**L, 2**L) @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
			else:
				U_tilde = A1
				for j in range(L//2):
					k_, l_ = (perms[k][2*j], perms[k][2*j+1])
					if z==j:
						U_tilde = applyG_tensor(Z.conj().T, U_tilde, k_, l_)
					else:
						U_tilde = applyG_tensor(Glist[k].conj().T, U_tilde, k_, l_)
				U_tilde = (A2.reshape(2**L, 2**L) @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
				U_tilde = applyG_block_tensor(Glist[k], U_tilde, L, perms[k])
				U_tilde = (A3.reshape(2**L, 2**L) @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
				U_tilde = (B.reshape(2**L, 2**L) @ applyG_block_tensor(Glist[i], U_tilde, L, 
					perms[i]).reshape(2**L, 2**L)).reshape([2]*2*L)


			for g in range(L//2):
				k__, l__ = (perms[i][2*g], perms[i][2*g+1])
				U_working = U_tilde.copy()

				for j in range(L//2):
					if j==g:
						continue
					else:
						k_, l_ = perms[i][2 * j], perms[i][2 * j + 1]
						U_working = applyG_tensor(Glist[i] if layer==0 else Glist[i].conj().T, 
							U_working, k_, l_)

				T = partial_trace_keep(U_working.reshape(2**L, 2**L), [k__, l__], L)
				if k__ > l__:
					SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
					T = SWAP @ T @ SWAP
				G += T if layer==0 else T.conj().T
	return G




def ansatz_grad_directed_dagger2(B1, B2, B3, A, Glist, Z, i, k, L, perms):
	# i < k
	G = np.zeros_like(Glist[i], dtype=complex)

	for layer in range(2):
		for z in range(L//2):
			# z in Z
			if layer==0:
				U_tilde = B1
				for j in range(L//2):
					k_, l_ = (perms[k][2*j], perms[k][2*j+1])
					if z==j:
						U_tilde = applyG_tensor(Z, U_tilde, k_, l_)
					else:
						U_tilde = applyG_tensor(Glist[k], U_tilde, k_, l_)
				U_tilde = (B2.reshape(2**L, 2**L) @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
				U_tilde = (B3.reshape(2**L, 2**L) @ applyG_block_tensor(Glist[k].conj().T, U_tilde, L, 
					perms[k]).reshape(2**L, 2**L)).reshape([2]*2*L)
				U_tilde = (A.reshape(2**L, 2**L) @ applyG_block_tensor(Glist[i].conj().T, U_tilde, L, 
					perms[i]).reshape(2**L, 2**L)).reshape([2]*2*L)
			else:
				U_tilde = (B1.reshape(2**L, 2**L) @ applyG_block_tensor(Glist[i], A, L, 
					perms[i]).reshape(2**L, 2**L)).reshape([2]*2*L)
				for j in range(L//2):
					k_, l_ = (perms[k][2*j], perms[k][2*j+1])
					if z==j:
						U_tilde = applyG_tensor(Z, U_tilde, k_, l_)
					else:
						U_tilde = applyG_tensor(Glist[k], U_tilde, k_, l_)
				U_tilde = (B2.reshape(2**L, 2**L) @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
				U_tilde = (B3.reshape(2**L, 2**L) @ applyG_block_tensor(Glist[k].conj().T, U_tilde, L, 
					perms[k]).reshape(2**L, 2**L)).reshape([2]*2*L)


			for g in range(L//2):
				k__, l__ = (perms[i][2*g], perms[i][2*g+1])
				U_working = U_tilde.copy()

				for j in range(L//2):
					if j==g:
						continue
					else:
						k_, l_ = perms[i][2 * j], perms[i][2 * j + 1]
						U_working = applyG_tensor(Glist[i] if layer==0 else Glist[i].conj().T, 
							U_working, k_, l_)

				T = partial_trace_keep(U_working.reshape(2**L, 2**L), [k__, l__], L)
				if k__ > l__:
					SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
					T = SWAP @ T @ SWAP
				G += T if layer==0 else T.conj().T


		for z in range(L//2):
			# z in Z^{\dag}
			if layer==0:
				U_tilde = (B2.reshape(2**L, 2**L)@applyG_block_tensor(Glist[k], B1, L, 
					perms[k]).reshape(2**L, 2**L)).reshape([2]*2*L)
				for j in range(L//2):
					k_, l_ = (perms[k][2*j], perms[k][2*j+1])
					if z==j:
						U_tilde = applyG_tensor(Z.conj().T, U_tilde, k_, l_)
					else:
						U_tilde = applyG_tensor(Glist[k].conj().T, U_tilde, k_, l_)
				U_tilde = (B3.reshape(2**L, 2**L) @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)
				U_tilde = (A.reshape(2**L, 2**L) @ applyG_block_tensor(Glist[i].conj().T, U_tilde, L, 
					perms[i]).reshape(2**L, 2**L)).reshape([2]*2*L)
			else:
				U_tilde = (B1.reshape(2**L, 2**L) @ applyG_block_tensor(Glist[i], A, L, 
					perms[i]).reshape(2**L, 2**L)).reshape([2]*2*L)
				U_tilde = (B2.reshape(2**L, 2**L)@applyG_block_tensor(Glist[k], U_tilde, L, 
					perms[k]).reshape(2**L, 2**L)).reshape([2]*2*L)
				for j in range(L//2):
					k_, l_ = (perms[k][2*j], perms[k][2*j+1])
					if z==j:
						U_tilde = applyG_tensor(Z.conj().T, U_tilde, k_, l_)
					else:
						U_tilde = applyG_tensor(Glist[k].conj().T, U_tilde, k_, l_)

				U_tilde = (B3.reshape(2**L, 2**L) @ U_tilde.reshape(2**L, 2**L)).reshape([2]*2*L)


			for g in range(L//2):
				k__, l__ = (perms[i][2*g], perms[i][2*g+1])
				U_working = U_tilde.copy()

				for j in range(L//2):
					if j==g:
						continue
					else:
						k_, l_ = perms[i][2 * j], perms[i][2 * j + 1]
						U_working = applyG_tensor(Glist[i] if layer==0 else Glist[i].conj().T, 
							U_working, k_, l_)

				T = partial_trace_keep(U_working.reshape(2**L, 2**L), [k__, l__], L)
				if k__ > l__:
					SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
					T = SWAP @ T @ SWAP
				G += T if layer==0 else T.conj().T
	return G


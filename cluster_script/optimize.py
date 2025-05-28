import qiskit
from qiskit.quantum_info import state_fidelity
import numpy as np
from numpy import linalg as LA
import qib
import matplotlib.pyplot as plt
import scipy

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


# Random unitary generator
def random_unitary(n):
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, _ = np.linalg.qr(A)
    return Q

L = 6
# construct Hamiltonian
latt = qib.lattice.IntegerLattice((L,), pbc=True)
field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
hamil = qib.HeisenbergHamiltonian(field, (1,1,1), (0,0,0)).as_matrix().toarray()

nlayers = 4
perms = [[i for i in range(L)] if i%2==0 else [i for i in range(1, L)]+[0] for i in range(nlayers)]


t = 1
U = scipy.linalg.expm(-1j*t*hamil)
U_back = scipy.linalg.expm(1j*t*hamil)
cU = U_back


import sys
sys.path.append("../ccU_tensor")
from optimize_tensor import optimize_circuit_tensor


Glists = []
f_iters = []
err_iters = []
best_f, best_f_index = 0, 0
for _ in range(20):
    Glist_start = [random_unitary(4) for _ in range(len(2*perms))]
    Glist, f_iter, err_iter = optimize_circuit_tensor(L, U, cU, Glist_start, perms, niter=80)
    print("Best f: ", f_iter[-1])
    print("Best err: ", err_iter[-1])

    Glists.append(Glist)
    f_iters.append(f_iter)
    err_iters.append(err_iter)

    if f_iter[-1] < best_f:
        best_f, best_f_index = f_iter[-1], _

    if f_iter[-1] <= -2**L + 1e-2:
        print("Converged to global optimum!")
        break


import h5py

with h5py.File(f"./results_data/Heisenberg1d_L{L}_t{t}_layers{len(perms)}_err{round(err_iters[best_f_index][-1], 2)}.hdf5", "w") as f:
    f.create_dataset("Glist", data=Glists[best_f_index])
    f.create_dataset("f_iter", data=f_iters[best_f_index])

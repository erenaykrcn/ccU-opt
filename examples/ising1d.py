import numpy as np
import scipy
import qib
import h5py
import rqcopt as oc
import matplotlib.pyplot as plt

import sys
sys.path.append("../ccU")
from optimize import optimize_circuit


ket_0 = np.array([[1],[0]])
ket_1 = np.array([[0],[1]])
rho_0_anc = ket_0 @ ket_0.T
rho_1_anc = ket_1 @ ket_1.T
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])



def ising1d_dynamics_opt(Glist_start, **kwargs):
    """
    Optimize the quantum gates in a brickwall layout to approximate
    the time evolution governed by an Ising Hamiltonian.
    """
    # side length of lattice
    L = 4
    # Hamiltonian parameters
    J = 1
    g = 1

    # construct Hamiltonian
    latt = qib.lattice.IntegerLattice((L,), pbc=True)
    field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
    hamil = qib.IsingHamiltonian(field, J, 0., g).as_matrix().todense()


    U = scipy.linalg.expm(-1j*hamil)
    U_back = scipy.linalg.expm(1j*hamil)
    cU = np.kron(rho_0_anc, U_back) + np.kron(rho_1_anc, U)

    perms = [[i for i in range(L)], [i for i in range(1, L)]+[0]]
    # perform optimization
    Vlist, f_iter, err_iter = optimize_circuit(L, U, cU, Glist_start, perms, **kwargs)

    # visualize optimization progress
    print(f"err_iter before: {err_iter[0]}")
    print(f"err_iter after {len(err_iter)-1} iterations: {err_iter[-1]}")
    plt.semilogy(range(len(err_iter)), err_iter)
    plt.xlabel("iteration")
    plt.ylabel("spectral norm error")
    plt.title(f"optimization progress for a quantum circuit with {len(Vlist)} layers")
    plt.show()
    # rescaled and shifted target function
    plt.semilogy(range(len(f_iter)), 1 + np.array(f_iter) / 2**L)
    plt.xlabel("iteration")
    plt.ylabel(r"$1 + f(\mathrm{Vlist})/2^L$")
    plt.title(f"optimization target function for a quantum circuit with {len(Vlist)} layers")
    plt.show()

    # save results to disk
    f_iter = np.array(f_iter)
    err_iter = np.array(err_iter)
    with h5py.File(f"ising1d_L{L}_dynamics_opt.hdf5", "w") as f:
        f.create_dataset("Vlist", data=Vlist)
        f.create_dataset("f_iter", data=f_iter)
        f.create_dataset("err_iter", data=err_iter)
        # store parameters
        f.attrs["L"] = L
        f.attrs["J"] = float(J)
        f.attrs["g"] = float(g)


def main():
    # Initial params.
    V1 = np.kron(Y, I2)
    V1 = np.kron(rho_1_anc, np.eye(4)) + np.kron(rho_0_anc, V1)
    V2 = np.kron(I2, I2)
    V2 = np.kron(rho_1_anc, np.eye(4)) + np.kron(rho_0_anc, V2)
    W1 = V1
    W2 = V2
    Glist_start = [V1, V2, W1, W2]
     
    ising1d_dynamics_opt(Glist_start, niter=10)

if __name__ == "__main__":
    main()


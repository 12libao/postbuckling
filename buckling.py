import json
import logging
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

import functools
import platform

import eigenvector_derivatives as eig_deriv
from eigenvector_derivatives import (IRAM, BasicLanczos, SpLuOperator,
                                     eval_adjoint_residual_norm)
from icecream import ic
from joblib import Parallel, delayed
from matplotlib.animation import FuncAnimation
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse import linalg

from fe_utils import populate_Be_and_Te, populate_nonlinear_strain_and_Be
from node_filter import NodeFilter

# Add the build directory to sys.path if the library is located there
build_dir = os.path.join(os.path.dirname(__file__), "build")
sys.path.append(build_dir)

import kokkos

if platform.system() == "Darwin":
    n_jobs = -1
else:
    n_jobs = 1


# Define a global switch for progress printing
PRINT_PROGRESS = True  # Set to False to disable progress printing
# ANSI escape code for green text
G = "\033[32m"
E = "\033[0m"

cw = plt.colormaps["coolwarm"]


def _print_progress(i, n, t0, prefix="", suffix="", bar_length=20):
    global PRINT_PROGRESS

    if not PRINT_PROGRESS:
        return

    # Calculate elapsed time and time remaining
    elapsed_time = time.time() - t0
    avg_time_per_iter = elapsed_time / (i + 1)
    tr = round(avg_time_per_iter * (n - i - 1))

    # Format time remaining in minutes and seconds
    if tr >= 60:
        minutes, seconds = divmod(tr, 60)
        t = f"{minutes}m {seconds}s"
    else:
        t = f"{tr}s"

    # Calculate percentage progress and construct progress bar
    percent = 100 * (i / (n - 1))  # Use n - 1 for zero-based indexing
    filled_length = int(bar_length * i // (n - 1))
    bar = "#" * filled_length + "-" * (bar_length - filled_length)

    # Print progress without adding newlines, padding to ensure consistency
    print(
        f"\r{G}{prefix}{E} --> |{bar}| {percent:.1f}%, Remain time: {t} {suffix}",
        end=" " * 2,
    )

    # Print newline only at the end when the iteration is complete
    if i == n - 1:
        # print(f"Total time: {G}{elapsed_time:.2f} s{E}")
        print(
            f"\r{G}{prefix}{E} --> |{bar}| {percent:.1f}%, Total time: {G}{elapsed_time:.2f} s{E}"
        )


# Timer decorator
def timeit(print_time=True):
    """
    Decorator to measure execution time of a function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)  # Call the actual function
            t1 = time.time()
            elapsed_time = t1 - t0

            if print_time:
                print(
                    f"{G}{func.__name__}{E} --> Total time: {G}{elapsed_time:.5f} s{E}"
                )

            return result

        return wrapper

    return decorator


class TopologyAnalysis:

    def __init__(
        self,
        fltr,
        conn,
        X,
        bcs,
        forces={},
        E=1.0,
        nu=0.3,
        ptype_K="simp",
        ptype_M="simp",
        ptype_G="simp",
        rho0_K=1e-6,
        rho0_M=1e-9,
        rho0_G=1e-6,
        p=3.0,
        q=5.0,
        density=1.0,
        sigma=3.0,
        N=10,
        m=None,
        solver_type="IRAM",
        tol=0.0,
        rtol=1e-10,
        eig_atol=1e-5,
        adjoint_method="shift-invert",
        adjoint_options={},
        cost=1,
        deriv_type="tensor",
    ):
        self.ptype_K = ptype_K.lower()
        self.ptype_M = ptype_M.lower()
        self.ptype_G = ptype_G.lower()

        self.rho0_K = rho0_K
        self.rho0_M = rho0_M
        self.rho0_G = rho0_G

        self.fltr = fltr
        self.conn = np.array(conn)
        self.X = np.array(X)
        self.p = p
        self.q = q
        self.density = density
        self.sigma = sigma  # Shift value
        self.N = N  # Number of modes
        self.m = m
        self.solver_type = solver_type
        self.tol = tol
        self.rtol = rtol
        self.eig_atol = eig_atol
        self.adjoint_method = adjoint_method
        self.adjoint_options = adjoint_options
        self.cost = cost
        self.deriv_type = deriv_type

        self.bcs = bcs
        self.forces = forces

        self.nelems = self.conn.shape[0]
        self.nnodes = int(np.max(self.conn)) + 1
        self.nvars = 2 * self.nnodes

        self.x = np.ones(self.fltr.num_design_vars)
        self.xb = np.zeros(self.x.shape)

        # # Set the initial design variable values
        # def _read_vtk(file, nx, ny):
        #     x, rho = [], []
        #     # nnodes = int((nx + 1) * (ny + 1))
        #     ic(nx, ny, self.nnodes)
        #     with open(file) as f:
        #         for num, line in enumerate(f, 1):
        #             if "design" in line:
        #                 x = np.loadtxt(file, skiprows=num + 1, max_rows=self.nnodes)
        #             if "rho" in line:
        #                 rho = np.loadtxt(file, skiprows=num + 1, max_rows=self.nnodes)
        #                 break

        #     return x, rho

        # self.x, self.rho = _read_vtk("./output/it_970.vtk", 300, 300)
        ic(self.x.shape, self.nnodes, self.nelems)

        self.Q = None
        self.lam = None

        # Compute the constitutive matrix
        self.E = E
        self.nu = nu
        self.C0 = E * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]
        )
        self.C0 *= 1.0 / (1.0 - nu**2)

        self.reduced = self._compute_reduced_variables(self.nvars, bcs)
        self.f = self._compute_forces(self.nvars, forces)
        self.bc_indices = self._get_BC_indices(bcs)

        # Set up the i-j indices for the matrix - these are the row
        # and column indices in the stiffness matrix
        self.var = np.zeros((self.conn.shape[0], 8), dtype=int)
        self.var[:, ::2] = 2 * self.conn
        self.var[:, 1::2] = 2 * self.conn + 1

        self.dfds = None
        self.pp = None

        i = []
        j = []
        for index in range(self.nelems):
            for ii in self.var[index, :]:
                for jj in self.var[index, :]:
                    i.append(ii)
                    j.append(jj)

        # Convert the lists into numpy arrays
        self.i = np.array(i, dtype=int)
        self.j = np.array(j, dtype=int)

        self.intital_Be_and_Te()
        self._init_profile()
        return

    def _compute_reduced_variables(self, nvars, bcs):
        """
        Compute the reduced set of variables
        """
        reduced = list(range(nvars))

        # For each node that is in the boundary condition dictionary
        for node in bcs:
            uv_list = bcs[node]

            # For each index in the boundary conditions (corresponding to
            # either a constraint on u and/or constraint on v
            for index in uv_list:
                var = 2 * node + index
                reduced.remove(var)

        return reduced

    def _compute_forces(self, nvars, forces):
        """
        Unpack the dictionary containing the forces
        """
        f = np.zeros(nvars)

        for node in forces:
            f[2 * node] += forces[node][0]
            f[2 * node + 1] += forces[node][1]

        return f

    def _get_BC_indices(self, bcs):
        """
        Get the indices of the boundary conditions
        """
        bc_indices = []

        for node in bcs:
            uv_list = bcs[node]

            for index in uv_list:
                var = 2 * node + index
                bc_indices.append(var)

        return bc_indices

    def get_stiffness_matrix(self, rho):
        """
        Assemble the stiffness matrix
        """
        rhoE = np.mean(rho[self.conn[:, :4]], axis=1)

        # Compute the element stiffnesses
        if self.ptype_K == "simp":
            C = np.outer(rhoE**self.p + self.rho0_K, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_K, self.C0)

        C = C.reshape((self.nelems, 3, 3))

        # Assemble all of the the 8 x 8 element stiffness matrix
        Ke = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)

        for i in range(4):
            Be = self.Be[:, :, :, i]
            detJ = self.detJ[:, i]
            Ke += detJ.reshape(-1, 1, 1) * Be.transpose(0, 2, 1) @ C @ Be

        K = sparse.csc_matrix((Ke.flatten(), (self.i, self.j)))  # column format

        return K

    def get_stress_stiffness_matrix(self, rho, u):
        """
        Assemble the stess stiffness matrix
        """
        # if any of rhoE or u1 is complex, set the dtype to complex
        if np.iscomplexobj(rho) or np.iscomplexobj(u):
            dt = complex
        else:
            dt = float

        rhoE = np.mean(rho[self.conn[:, :4]], axis=1)
        # Get the element-wise solution variables
        ue = np.zeros((self.nelems, 8), dtype=dt)
        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        # Compute the element stiffnesses
        if self.ptype_G == "simp":
            C = np.outer(rhoE**self.p + self.rho0_G, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_G, self.C0)

        C = C.reshape((self.nelems, 3, 3))

        Ge = np.zeros((self.nelems, 8, 8), dtype=dt)

        for i in range(4):
            detJ = self.detJ[:, i]
            Be = self.Be[:, :, :, i]
            Te = self.Te[:, :, :, :, i]

            # Compute the stresses in each element
            CBe = C @ Be
            s = np.einsum("nik,nk -> ni", CBe, ue)
            G0e = detJ.reshape(-1, 1, 1) * np.einsum("ni,nijl -> njl", s, Te)

            Ge[:, 0::2, 0::2] += G0e
            Ge[:, 1::2, 1::2] += G0e

        G = sparse.csc_matrix((Ge.flatten(), (self.i, self.j)))

        return G

    # def get_stiffness_matrix(self, rho):
    #     """
    #     Assemble the stiffness matrix
    #     """

    #     Ke = kokkos.assemble_stiffness_matrix(
    #         rho,
    #         self.detJ_kokkos,
    #         self.Be_kokkos,
    #         self.conn,
    #         self.C0,
    #         self.rho0_K,
    #         self.ptype_K,
    #         self.p,
    #         self.q,
    #     )
    #     K = sparse.csc_matrix((Ke.flatten(), (self.i, self.j)))

    #     return K

    # def get_stress_stiffness_matrix(self, rho, u):
    #     """
    #     Assemble the stess stiffness matrix
    #     """
    #     Ge = kokkos.assemble_stress_stiffness(
    #         rho,
    #         u,
    #         self.detJ_kokkos,
    #         self.Be_kokkos,
    #         self.Te_kokkos,
    #         self.conn,
    #         self.C0,
    #         self.rho0_G,
    #         self.ptype_K,
    #         self.p,
    #         self.q,
    #     )
    #     G = sparse.csc_matrix((Ge.flatten(), (self.i, self.j)))

    #     return G

    def get_stiffness_matrix_deriv(self, rhoE, psi, u):
        """
        Compute the derivative of the stiffness matrix times the vectors psi and u
        """

        dfdrhoE = np.zeros(self.nelems)

        # The element-wise variables
        ue = np.zeros((self.nelems, 8) + u.shape[1:])
        psie = np.zeros((self.nelems, 8) + psi.shape[1:])

        ue[:, ::2, ...] = u[2 * self.conn, ...]
        ue[:, 1::2, ...] = u[2 * self.conn + 1, ...]

        psie[:, ::2, ...] = psi[2 * self.conn, ...]
        psie[:, 1::2, ...] = psi[2 * self.conn + 1, ...]

        for i in range(4):
            Be = self.Be[:, :, :, i]
            detJ = self.detJ[:, i]

            if psi.ndim == 1 and u.ndim == 1:
                se = np.einsum("nij,nj -> ni", Be, psie)
                te = np.einsum("nij,nj -> ni", Be, ue)

                Cte = np.einsum("ij,nj -> ni", self.C0, te)
                dfdrhoE += detJ * np.einsum("ni,ni -> n", se, Cte)

            elif psi.ndim == 2 and u.ndim == 2:
                se = Be @ psie
                te = Be @ ue
                dfdrhoE += detJ * np.einsum("ij,njk,nik -> n", self.C0, se, te)

        if self.ptype_K == "simp":
            dfdrhoE[:] *= self.p * rhoE ** (self.p - 1.0)
        else:  # ramp
            dfdrhoE[:] *= (1.0 + self.q) / (1.0 + self.q * (1.0 - rhoE)) ** 2

        dKdrho = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(dKdrho, self.conn[:, i], dfdrhoE)
        dKdrho *= 0.25

        return self.fltr.apply_gradient(dKdrho, self.x)

    def get_stress_stiffness_matrix_xuderiv(self, rhoE, u0, u, v):
        dGdx = self.get_stress_stiffness_matrix_xderiv(rhoE, u0, u, v)
        dGdu = self.get_stress_stiffness_matrix_uderiv(rhoE, u, v)
        dGdur = self.reduce_vector(dGdu)
        adjr = -self.Kfact(dGdur)
        adj = self.full_vector(adjr)
        dGdx += self.get_stiffness_matrix_deriv(rhoE, adj, u0)

        return dGdx

    def get_dK1dx(self, rhoE, u1, psi, u):
        """
        Compute the derivative of the stiffness matrix times the vectors psi and u
        """

        df1drhoE = np.zeros(self.nelems)

        u1 = u1 / np.linalg.norm(u1)

        # The element-wise variables
        ue = np.zeros((self.nelems, 8), dtype=rhoE.dtype)
        psie = np.zeros((self.nelems, 8), dtype=rhoE.dtype)
        ue1 = np.zeros((self.nelems, 8), dtype=rhoE.dtype)

        ue[:, ::2, ...] = u[2 * self.conn, ...]
        ue[:, 1::2, ...] = u[2 * self.conn + 1, ...]

        psie[:, ::2, ...] = psi[2 * self.conn, ...]
        psie[:, 1::2, ...] = psi[2 * self.conn + 1, ...]

        ue1[:, ::2, ...] = u1[2 * self.conn, ...]
        ue1[:, 1::2, ...] = u1[2 * self.conn + 1, ...]

        for i in range(4):
            Be = self.Be[:, :, :, i]
            detJ = self.detJ[:, i]

            Be1 = np.zeros((self.nelems, 3, 8))
            populate_nonlinear_strain_and_Be(Be, ue1, Be1)

            se = np.einsum("nij,nj -> ni", Be, psie)
            te = np.einsum("nij,nj -> ni", Be1, ue)

            Cte = np.einsum("ij,nj -> ni", self.C0, te)
            df1drhoE += detJ * np.einsum("ni,ni -> n", se, Cte)

        if self.ptype_K == "simp":
            df1drhoE[:] *= self.p * rhoE ** (self.p - 1.0)
        else:  # ramp
            df1drhoE[:] *= (1.0 + self.q) / (1.0 + self.q * (1.0 - rhoE)) ** 2

        dK1drho = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(dK1drho, self.conn[:, i], df1drhoE)
        dK1drho *= 0.25

        return self.fltr.apply_gradient(dK1drho, self.x)

    def get_dK11dx(self, rhoE, u1, psi, u):
        """
        Compute the derivative of the stiffness matrix times the vectors psi and u
        """

        df11drhoE = np.zeros(self.nelems)

        u1 = u1 / np.linalg.norm(u1)

        # The element-wise variables
        ue = np.zeros((self.nelems, 8), dtype=rhoE.dtype)
        psie = np.zeros((self.nelems, 8), dtype=rhoE.dtype)
        ue1 = np.zeros((self.nelems, 8), dtype=rhoE.dtype)

        ue[:, ::2, ...] = u[2 * self.conn, ...]
        ue[:, 1::2, ...] = u[2 * self.conn + 1, ...]

        psie[:, ::2, ...] = psi[2 * self.conn, ...]
        psie[:, 1::2, ...] = psi[2 * self.conn + 1, ...]

        ue1[:, ::2, ...] = u1[2 * self.conn, ...]
        ue1[:, 1::2, ...] = u1[2 * self.conn + 1, ...]

        for i in range(4):
            Be = self.Be[:, :, :, i]
            detJ = self.detJ[:, i]

            Be1 = np.zeros((self.nelems, 3, 8))
            populate_nonlinear_strain_and_Be(Be, ue1, Be1)

            se = np.einsum("nij,nj -> ni", Be1, psie)
            te = np.einsum("nij,nj -> ni", Be1, ue)

            Cte = np.einsum("ij,nj -> ni", self.C0, te)
            df11drhoE += detJ * np.einsum("ni,ni -> n", se, Cte)

        if self.ptype_K == "simp":
            df11drhoE[:] *= self.p * rhoE ** (self.p - 1.0)
        else:  # ramp
            df11drhoE[:] *= (1.0 + self.q) / (1.0 + self.q * (1.0 - rhoE)) ** 2

        dK11drho = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(dK11drho, self.conn[:, i], df11drhoE)
        dK11drho *= 0.25

        return self.fltr.apply_gradient(dK11drho, self.x)

    def get_dK1dx_cd(self, x, dQdx, Q0, u, v, dh=1e-4):
        """
        Compute the derivative of the stiffness matrix times the vectors u and v using central differences
        """

        # Precompute detJ * Be terms outside the loop to avoid redundancy
        detJBe = np.zeros((self.nelems, 8, 3, 4))
        for i in range(4):
            Be = self.Be[:, :, :, i]
            detJ = self.detJ[:, i]
            detJBe[:, :, :, i] = detJ.reshape(-1, 1, 1) * Be.transpose(0, 2, 1)

        conn_indices = self.conn[:, :4]
        dK1dx = np.zeros(self.nnodes)

        t0 = time.time()
        for n in range(x.size):
            # Add a small perturbation to the design variable
            x1, x2 = x.copy(), x.copy()
            x1[n] += dh
            x2[n] -= dh

            # Apply filtering and compute average rhoE for both perturbed vectors
            rhoE1 = np.mean(self.fltr.apply(x1)[conn_indices], axis=1)
            rhoE2 = np.mean(self.fltr.apply(x2)[conn_indices], axis=1)

            # Compute Q1 and Q2 using dQdx and normalize them
            Q1 = dQdx @ (x1 - x) + Q0
            Q2 = dQdx @ (x2 - x) + Q0

            # Adjust the direction of Q1 and Q2 if necessary
            Q1 = np.where(np.dot(Q0, Q1) < 0, -Q1, Q1)
            Q2 = np.where(np.dot(Q0, Q2) < 0, -Q2, Q2)

            # Normalize Q1 and Q2
            Q1 /= np.linalg.norm(Q1)
            Q2 /= np.linalg.norm(Q2)

            ue1 = np.zeros((self.nelems, 8), dtype=rhoE1.dtype)
            ue2 = np.zeros((self.nelems, 8), dtype=rhoE2.dtype)
            ue1[:, ::2] = Q1[2 * self.conn]
            ue1[:, 1::2] = Q1[2 * self.conn + 1]
            ue2[:, ::2] = Q2[2 * self.conn]
            ue2[:, 1::2] = Q2[2 * self.conn + 1]

            # Compute the element stiffnesses
            if self.ptype_K == "simp":
                C1 = np.outer(rhoE1**self.p + self.rho0_K, self.C0)
                C2 = np.outer(rhoE2**self.p + self.rho0_K, self.C0)
            else:  # ramp
                C1 = np.outer(
                    rhoE1 / (1.0 + self.q * (1.0 - rhoE1)) + self.rho0_K, self.C0
                )
                C2 = np.outer(
                    rhoE2 / (1.0 + self.q * (1.0 - rhoE2)) + self.rho0_K, self.C0
                )

            C1 = C1.reshape((self.nelems, 3, 3))
            C2 = C2.reshape((self.nelems, 3, 3))

            # Compute dKe and sum up contributions from all elements
            dKe = np.zeros((self.nelems, 8, 8), dtype=rhoE1.dtype)
            for i in range(4):
                Be = self.Be[:, :, :, i]
                detJBei = detJBe[:, :, :, i]

                Be1 = np.zeros((self.nelems, 3, 8), dtype=rhoE1.dtype)
                Be2 = np.zeros((self.nelems, 3, 8), dtype=rhoE2.dtype)
                populate_nonlinear_strain_and_Be(Be, ue1, Be1)
                populate_nonlinear_strain_and_Be(Be, ue2, Be2)

                dKe += detJBei @ (C1 @ Be1 - C2 @ Be2)

            dK = sparse.csc_matrix((dKe.flatten(), (self.i, self.j)))
            dK1dx[n] = (u.T @ dK @ v) / (2 * dh)

            _print_progress(n, self.nnodes, t0, "dK1/dx")

        return dK1dx

    def setBCs(self, u):
        """
        Set entry of the vector u corresponding to the boundary conditions to zero
        """
        u[self.bc_indices] = 0.0
        return

    def getEnergy(self, rhoE, u, lam):
        # Compute the element stiffnesses
        if self.ptype_K == "simp":
            CK = np.outer(rhoE**self.p + self.rho0_K, self.C0)
            CG = np.outer(rhoE**self.p + self.rho0_G, self.C0)
        else:  # ramp
            CK = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_K, self.C0)
            CG = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_G, self.C0)

        CK = CK.reshape((self.nelems, 3, 3))
        CG = CG.reshape((self.nelems, 3, 3))

        ue = np.zeros((self.nelems, 8))
        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        energy_li = 0.0
        energy_nl = 0.0
        for i in range(4):
            detJ = self.detJ[:, i]
            Be = self.Be[:, :, :, i]

            strain_li = np.zeros((self.nelems, 3))
            strain_nl = np.zeros((self.nelems, 3))
            Be1 = np.zeros((self.nelems, 3, 8))
            populate_nonlinear_strain_and_Be(Be, ue, Be1, strain_nl, strain_li)

            # compute the energy
            # where strain_li = Be @ ue, strain_nl /= (Be + Be1) @ ue
            for n in range(self.nelems):
                energy_li += 0.5 * detJ[n] * strain_li[n].T @ CK[n] @ strain_li[n]
                energy_nl += 0.5 * detJ[n] * strain_nl[n].T @ CK[n] @ strain_nl[n]

        # where f is with negative sign, so we add it
        energy_li = energy_li - lam * self.f.T @ u
        energy_nl = energy_nl - lam * self.f.T @ u

        return energy_li, energy_nl

    def getResidual(self, rhoE, u, lam):
        # Compute the element stiffnesses
        if self.ptype_K == "simp":
            CK = np.outer(rhoE**self.p + self.rho0_K, self.C0)
            CG = np.outer(rhoE**self.p + self.rho0_G, self.C0)
        else:  # ramp
            CK = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_K, self.C0)
            CG = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_G, self.C0)

        CK = CK.reshape((self.nelems, 3, 3))
        CG = CG.reshape((self.nelems, 3, 3))

        ue = np.zeros((self.nelems, 8))
        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        R_li = np.zeros(u.shape)
        R_nl = np.zeros(u.shape)

        for i in range(4):
            detJ = self.detJ[:, i]
            Be = self.Be[:, :, :, i]

            strain_li = np.zeros((self.nelems, 3))
            strain_nl = np.zeros((self.nelems, 3))
            Be1 = np.zeros((self.nelems, 3, 8))
            populate_nonlinear_strain_and_Be(Be, ue, Be1, strain_nl, strain_li)

            for n in range(self.nelems):
                Re_li = detJ[n] * strain_li[n].T @ CK[n] @ Be[n]
                Re_nl = detJ[n] * strain_nl[n].T @ CK[n] @ (Be[n] + Be1[n])

                np.add.at(R_li, 2 * self.conn[n], Re_li[::2])
                np.add.at(R_li, 2 * self.conn[n] + 1, Re_li[1::2])
                np.add.at(R_nl, 2 * self.conn[n], Re_nl[::2])
                np.add.at(R_nl, 2 * self.conn[n] + 1, Re_nl[1::2])

        # where f is with negative sign, so we add it
        R_li = R_li - lam * self.f
        R_nl = R_nl - lam * self.f

        self.setBCs(R_li)
        self.setBCs(R_nl)

        return R_li, R_nl

    def getKt(self, rhoE, u, return_G=False):
        # Compute the element stiffnesses
        if self.ptype_K == "simp":
            CK = np.outer(rhoE**self.p + self.rho0_K, self.C0)
            CG = np.outer(rhoE**self.p + self.rho0_G, self.C0)
        else:  # ramp
            CK = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_K, self.C0)
            CG = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_G, self.C0)

        CK = CK.reshape((self.nelems, 3, 3))
        CG = CG.reshape((self.nelems, 3, 3))

        ue = np.zeros((self.nelems, 8))
        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        Ke_li = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)
        Ke_nl = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)
        Ge_nl = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)

        for i in range(4):
            detJ = self.detJ[:, i]
            Be = self.Be[:, :, :, i]
            Te = self.Te[:, :, :, :, i]

            strain_li = np.zeros((self.nelems, 3))
            strain_nl = np.zeros((self.nelems, 3))
            Be1 = np.zeros((self.nelems, 3, 8))
            populate_nonlinear_strain_and_Be(Be, ue, Be1, strain_nl, strain_li)

            Be_nl = Be + Be1
            Ke_li += detJ.reshape(-1, 1, 1) * Be.transpose(0, 2, 1) @ CK @ Be
            Ke_nl += detJ.reshape(-1, 1, 1) * Be_nl.transpose(0, 2, 1) @ CK @ Be_nl

            s = np.einsum("nij,nj -> ni", CK, strain_nl)
            G0e_nl = np.einsum("n,ni,nijl -> njl", detJ, s, Te)

            Ge_nl[:, 0::2, 0::2] += G0e_nl
            Ge_nl[:, 1::2, 1::2] += G0e_nl

            # for n in range(self.nelems):
            #     Ke_li[n] += detJ[n] * Be[n].T @ CK[n] @ Be[n]

            #     Ke_nl[n] += detJ[n] * (Be[n] + Be1[n]).T @ CK[n] @ (Be[n] + Be1[n])
            #     G0e_nl = np.einsum(
            #         "i,ijl -> jl", detJ[n] * strain_nl[n].T @ CG[n], Te[n]
            #     )
            #     Ge_nl[n, 0::2, 0::2] += G0e_nl
            #     Ge_nl[n, 1::2, 1::2] += G0e_nl

        if return_G:
            K_nl = sparse.csc_matrix(
                (Ke_nl.ravel(), (self.i, self.j)), shape=(self.nvars, self.nvars)
            )
            G_nl = sparse.csc_matrix(
                (Ge_nl.ravel(), (self.i, self.j)), shape=(self.nvars, self.nvars)
            )

            # Apply boundary conditions for Kt_nl
            K_nl[self.bc_indices, :] = 0
            K_nl[:, self.bc_indices] = 0
            K_nl[self.bc_indices, self.bc_indices] = 1

            G_nl[self.bc_indices, :] = 0
            G_nl[:, self.bc_indices] = 0
            G_nl[self.bc_indices, self.bc_indices] = 1

            return K_nl, G_nl

        Kt_li = sparse.csc_matrix(
            (Ke_li.ravel(), (self.i, self.j)), shape=(self.nvars, self.nvars)
        )
        Kt_nl = sparse.csc_matrix(
            ((Ke_nl + Ge_nl).ravel(), (self.i, self.j)), shape=(self.nvars, self.nvars)
        )

        # Apply boundary conditions for Kt_li
        Kt_li[self.bc_indices, :] = 0
        Kt_li[:, self.bc_indices] = 0
        Kt_li[self.bc_indices, self.bc_indices] = 1

        Kt_nl[self.bc_indices, :] = 0
        Kt_nl[:, self.bc_indices] = 0
        Kt_nl[self.bc_indices, self.bc_indices] = 1

        return Kt_li, Kt_nl

    def getGt(self, rhoE, u, v, dh=1e-4):
        """
        Compute directional derivative of the tangent stiffness matrix use central difference
        G = d(Kt(u + epsilon * v)) / d(epsilon)
        """
        Kt1 = self.getKt(rhoE, u + dh * v)[1]
        Kt2 = self.getKt(rhoE, u - dh * v)[1]
        G = (Kt1 - Kt2) / (2 * dh)
        return G

    def getGtprod(self, rhoE, u, v, p, dh=1e-4):
        """
        Compute: d(Kt(u + epsilon * v)) / d(epsilon) * w
        """
        G = self.getGt(rhoE, u, v, dh)
        return G @ p

    def getHprod(self, rhoE, u, v, w, p, dh=1e-4):
        """
        Compute: d(Kt(u + epsilon * w; v)) * p
        """
        Gt1 = self.getGtprod(rhoE, u + dh * w, v, p)
        Gt2 = self.getGtprod(rhoE, u - dh * w, v, p)
        Hprod = (Gt1 - Gt2) / (2 * dh)
        return Hprod

    def buckling_analysis(self, rhoE, f, lam0, u0):
        Kt = self.getKt(rhoE, u0)[1]
        u1 = sparse.linalg.spsolve(Kt, f)
        Gt = self.getGt(rhoE, u0, u1)

        Ktr = self.reduce_matrix(Kt)
        Gtr = self.reduce_matrix(Gt)

        # solve the eigenvalue problem
        mu, vec = sparse.linalg.eigsh(Gtr, M=Ktr, k=1, which="SM", sigma=0.1)
        lam_c = -1.0 / mu[0][0]

        return lam0 + lam_c, vec

    def compute_path(self, rhoE, f, u0=None, lam1=1.8, lam2=-1.0, lam3=-10.0, dh=1e-4):
        if u0 is None:
            u0 = np.zeros(2 * self.nnodes)

        fr = self.reduce_vector(f)

        # solve for u1
        Kt = self.getKt(rhoE, u0)[1]
        Ktr = self.reduce_matrix(Kt)
        u1r = sparse.linalg.spsolve(Ktr, lam1 * fr)
        u1 = self.full_vector(u1r)

        # solve for u2
        Gp = self.getGtprod(rhoE, u0, u1, u1, dh)
        Gpr = self.reduce_vector(Gp)
        u2r = sparse.linalg.spsolve(Ktr, lam2 * fr - Gpr)
        u2 = self.full_vector(u2r)

        # solve for u3
        Gp = self.getGtprod(rhoE, u0, u1, u2, dh)
        Gpr = self.reduce_vector(Gp)
        Hp = self.getHprod(rhoE, u0, u1, u1, u1, dh)
        Hpr = self.reduce_vector(Hp)
        u3r = sparse.linalg.spsolve(Ktr, lam3 * fr - 3.0 * Gpr - Hpr)
        u3 = self.full_vector(u3r)

        return u0, u1, u2, u3

    def testR(self, u=None, lam=1.0, p=None, dh=1e-4):
        # check the residual use finite difference
        if u is None:
            u = np.random.rand(2 * self.nnodes)
        if p is None:
            p = np.random.rand(2 * self.nnodes)

        self.setBCs(u)
        self.setBCs(p)

        R_li, R_nl = self.getResidual(self.rhoE, u, lam)
        ans_li = np.dot(R_li, p)
        ans_nl = np.dot(R_nl, p)

        cd_li = (
            self.getEnergy(self.rhoE, u + dh * p, lam)[0]
            - self.getEnergy(self.rhoE, u - dh * p, lam)[0]
        ) / (2 * dh)
        cd_nl = (
            self.getEnergy(self.rhoE, u + dh * p, lam)[1]
            - self.getEnergy(self.rhoE, u - dh * p, lam)[1]
        ) / (2 * dh)

        print(
            "Residual linear:    ans: %10.5e,  cd: %10.5e,  rel.err: %10.5e"
            % (ans_li, cd_li, (ans_li - cd_li) / cd_li)
        )
        print(
            "Residual nonlinear: ans: %10.5e,  cd: %10.5e,  rel.err: %10.5e"
            % (ans_nl, cd_nl, (ans_nl - cd_nl) / cd_nl),
            "\n",
        )
        return

    def testKt(self, u=None, lam=1.0, p=None, q=None, dh=1e-4):
        # check the tangent stiffness use finite difference
        if u is None:
            u = np.random.rand(2 * self.nnodes)
        if p is None:
            p = np.random.rand(2 * self.nnodes)
        if q is None:
            q = np.random.rand(2 * self.nnodes)

        self.setBCs(u)
        self.setBCs(p)
        self.setBCs(q)

        Kt_li, Kt_nl = self.getKt(self.rhoE, u)
        R_li_1, R_nl_1 = self.getResidual(self.rhoE, u + dh * p, lam)
        R_li_2, R_nl_2 = self.getResidual(self.rhoE, u - dh * p, lam)

        ans_li = np.dot(Kt_li @ p, q)
        ans_nl = np.dot(Kt_nl @ p, q)

        cd_li = np.dot(R_li_1 - R_li_2, q) / (2 * dh)
        cd_nl = np.dot(R_nl_1 - R_nl_2, q) / (2 * dh)

        print(
            "Kt linear:    ans: %10.5e,  cd: %10.5e,  rel.err: %10.5e"
            % (ans_li, cd_li, (ans_li - cd_li) / cd_li)
        )
        print(
            "Kt nonlinear: ans: %10.5e,  cd: %10.5e,  rel.err: %10.5e"
            % (ans_nl, cd_nl, (ans_nl - cd_nl) / cd_nl),
            "\n",
        )
        return

    def newton_raphson(self, lam, u=None, tol=1e-12, maxiter=100):
        if u is None:
            u = np.zeros(2 * self.nnodes)

        self.setBCs(u)

        # arch-length algorithm
        for i in range(maxiter):
            R = self.getResidual(self.rhoE, u, lam)[1]

            # check the convergence
            rnorm = np.linalg.norm(R)
            print(f"Newton-Raphson[{i:3d}]  {rnorm:15.10e}")
            if rnorm < tol:
                break

            Kt = self.getKt(self.rhoE, u)[1]
            Ktr = self.reduce_matrix(Kt)
            Rr = self.reduce_vector(R)

            # solve the linear system to get du, and u = u + du
            ur = sparse.linalg.spsolve(Ktr, Rr)
            u -= self.full_vector(ur)

        return u

    def arc_length_method(
        self,
        Dl=1.0,
        scale=1.0,
        u=None,
        tol=1e-12,
        maxiter=100,
        k_max=10,
        lmax=None,
        geteigval=False,
        u_imp=None,
    ):
        if u is None:
            u = np.zeros(2 * self.nnodes)

        if u_imp is not None:
            self.X[:, 0] += u_imp[::2]
            self.X[:, 1] += u_imp[1::2]
            self.intital_Be_and_Te()

        self.setBCs(u)
        u_prev = u.copy()
        u_prev_prev = u.copy()

        l, l_prev, l_prev_prev = Dl, 0, 0
        Ds, Ds_prev, Ds_max, Ds_min = Dl, Dl, Dl / scale, Dl / 1024

        u_list = [u_prev]
        l_list = [l_prev]
        eig_Kt_list = []
        if geteigval:
            Kt = self.getKt(self.rhoE, u)[1]
            Ktr = self.reduce_matrix(Kt)
            eig_Kt = sparse.linalg.eigsh(
                Ktr, k=1, which="LM", return_eigenvectors=False, sigma=0.0
            )
            eig_Kt_list.append(eig_Kt)

        converged = False
        converged_prev = False

        ff = np.dot(self.f, self.f)
        fr = self.reduce_vector(self.f).reshape(-1, 1)

        # arch-length algorithm
        for n in range(maxiter):
            if n > 0:
                a0 = Ds / Ds_prev
                u = (1 + a0) * u_prev - a0 * u_prev_prev
                l = (1 + a0) * l_prev - a0 * l_prev_prev

            Du = u - u_prev
            Dl = l - l_prev

            converged_prev = converged
            converged = False

            print(
                f"\nArch-Length[{n:3d}]  lam: {l:5.2f}, D_lam: {Ds:5.2e}, D_max: {Ds_max:5.2e}"
            )

            for k in range(k_max):
                # compute the residual and tangent stiffness
                R = self.getResidual(self.rhoE, u, l)[1]
                Kt = self.getKt(self.rhoE, u)[1]

                if n == 0:
                    A = 0.0
                    a = np.zeros_like(Du)
                    b = 1.0
                else:
                    A = np.dot(Du, Du) + Dl**2 * ff - Ds**2
                    a = 2 * Du
                    b = 2 * Dl * ff

                res = np.sqrt(np.linalg.norm(R) ** 2 + A**2)
                print(f"    {k:3d}  {res:15.10e}")

                if res < tol:
                    converged = True
                    break

                # apply the boundary conditions
                Ktr = self.reduce_matrix(Kt)
                Rr = self.reduce_vector(R).reshape(-1, 1)
                ar = self.reduce_vector(a).reshape(-1, 1)
                b = sparse.csc_matrix(b).reshape(-1, 1)

                # construct the matrix and solve the linear system
                mat = sparse.bmat([[Ktr, -fr], [ar.T, b]], format="csc")
                dx = sparse.linalg.spsolve(mat, -np.vstack([Rr, A]))
                du = self.full_vector(dx[:-1])
                dl = dx[-1]

                # update the solution
                u += du
                l += dl
                Du += du
                Dl += dl

            if converged:
                if n == 0:
                    Ds = np.sqrt(np.dot(Du.T, Du) + Dl**2 * ff)
                    Ds_min, Ds_max = Ds / 1024, Ds / scale

                l_prev_prev = l_prev
                l_prev = l
                u_prev_prev = u_prev
                u_prev = u
                Ds_prev = Ds

                if converged_prev:
                    Ds = min(max(2 * Ds, Ds_min), Ds_max)

            else:
                if converged_prev:
                    Ds = max(Ds / (2 * scale), Ds_min)
                else:
                    Ds = max(Ds / (4 * scale), Ds_min)

            if lmax is not None and abs(l) > lmax:
                break

            u_list.append(u)
            l_list.append(l)

            # compute the eigenvalue for the tangent stiffness
            if geteigval:
                eig_Kt = sparse.linalg.eigsh(
                    Ktr, k=1, which="LM", return_eigenvectors=False, sigma=0.0
                )
                eig_Kt_list.append(eig_Kt)

        if u_imp is not None:
            self.X[:, 0] -= u_imp[::2]
            self.X[:, 1] -= u_imp[1::2]
            self.intital_Be_and_Te()

        if geteigval:
            return u_list, l_list, eig_Kt_list
        else:
            return u_list, l_list

    def get_K1(self, rhoE, u1):
        """
        u1: the critical buckling mode (without normalization)
        """

        # if any of rhoE or u1 is complex, set the dtype to complex
        if np.iscomplexobj(rhoE) or np.iscomplexobj(u1):
            dt = complex
        else:
            dt = float

        ue1 = np.zeros((self.nelems, 8), dtype=dt)
        ue1[:, ::2] = u1[2 * self.conn]
        ue1[:, 1::2] = u1[2 * self.conn + 1]

        # Compute the element stiffnesses
        if self.ptype_K == "simp":
            C = np.outer(rhoE**self.p + self.rho0_K, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_K, self.C0)

        C = C.reshape((self.nelems, 3, 3))

        Ke1 = np.zeros((self.nelems, 8, 8), dtype=dt)

        for i in range(4):
            detJ = self.detJ[:, i]
            Be = self.Be[:, :, :, i]

            Be1 = np.zeros((self.nelems, 3, 8), dtype=dt)
            populate_nonlinear_strain_and_Be(Be, ue1, Be1)

            CBe1 = C @ Be1
            Ke1 += detJ.reshape(-1, 1, 1) * Be.transpose(0, 2, 1) @ CBe1

        K1 = sparse.csc_matrix((Ke1.flatten(), (self.i, self.j)))

        return K1

    def get_K11(self, rhoE, u1):
        """
        u1: the critical buckling mode (without normalization)
        """

        # if any of rhoE or u1 is complex, set the dtype to complex
        if np.iscomplexobj(rhoE) or np.iscomplexobj(u1):
            dt = complex
        else:
            dt = float

        ue1 = np.zeros((self.nelems, 8), dtype=dt)
        ue1[:, ::2] = u1[2 * self.conn]
        ue1[:, 1::2] = u1[2 * self.conn + 1]

        # Compute the element stiffnesses
        if self.ptype_K == "simp":
            C = np.outer(rhoE**self.p + self.rho0_K, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_K, self.C0)

        C = C.reshape((self.nelems, 3, 3))

        Ke11 = np.zeros((self.nelems, 8, 8), dtype=dt)

        for i in range(4):
            detJ = self.detJ[:, i]
            Be = self.Be[:, :, :, i]

            Be1 = np.zeros((self.nelems, 3, 8), dtype=dt)
            populate_nonlinear_strain_and_Be(Be, ue1, Be1)

            CBe1 = C @ Be1
            Ke11 += detJ.reshape(-1, 1, 1) * Be1.transpose(0, 2, 1) @ CBe1

        K11 = sparse.csc_matrix((Ke11.flatten(), (self.i, self.j)))

        return K11

    def get_G1(self, rhoE, u1, norm=True):
        """
        u1: the critical buckling mode (without normalization)
        """
        u1_norm = np.linalg.norm(u1)

        if norm:
            u1 = u1 / u1_norm

        ue1 = np.zeros((self.nelems, 8), dtype=rhoE.dtype)
        ue1[:, ::2, ...] = u1[2 * self.conn, ...]
        ue1[:, 1::2, ...] = u1[2 * self.conn + 1, ...]

        # Compute the element stiffnesses
        if self.ptype_K == "simp":
            CK = np.outer(rhoE**self.p + self.rho0_K, self.C0)
        else:  # ramp
            CK = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_K, self.C0)

        CK = CK.reshape((self.nelems, 3, 3))

        Ge1 = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)

        for i in range(4):
            detJ = self.detJ[:, i]
            Be = self.Be[:, :, :, i]
            Te = self.Te[:, :, :, :, i]

            CKBe = CK @ Be

            s1 = np.einsum("nik,nk -> ni", CKBe, ue1)
            G0e1 = np.einsum("n,ni,nijl -> njl", detJ, s1, Te)
            Ge1[:, 0::2, 0::2] += G0e1
            Ge1[:, 1::2, 1::2] += G0e1

        # Create sparse matrices directly
        G1 = sparse.csc_matrix((Ge1.flatten(), (self.i, self.j)))

        return G1

    def get_koiter_a(self, rhoE, u1, Q0_norm=None):
        """
        u1: the critical buckling mode (without normalization)
        """
        if Q0_norm is None:
            Q0_norm = np.linalg.norm(u1)

        u1 = u1 / Q0_norm

        #  if any of rhoE or u1 is complex, set the dtype to complex
        if np.iscomplexobj(rhoE) or np.iscomplexobj(u1):
            dt = complex
        else:
            dt = float

        ue1 = np.zeros((self.nelems, 8), dtype=dt)
        ue1[:, ::2] = u1[2 * self.conn]
        ue1[:, 1::2] = u1[2 * self.conn + 1]

        # Compute the element stiffnesses
        if self.ptype_K == "simp":
            CK = np.outer(rhoE**self.p + self.rho0_K, self.C0)
        else:  # ramp
            CK = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_K, self.C0)

        CK = CK.reshape((self.nelems, 3, 3))
        Ke1 = np.zeros((self.nelems, 8, 8), dtype=dt)

        for i in range(4):
            detJ = self.detJ[:, i]
            Be = self.Be[:, :, :, i]

            Be1 = np.zeros((self.nelems, 3, 8), dtype=dt)
            populate_nonlinear_strain_and_Be(Be, ue1, Be1)

            Ke1 += detJ.reshape(-1, 1, 1) * Be.transpose(0, 2, 1) @ CK @ Be1

        # Create sparse matrices directly
        self.K1 = sparse.csc_matrix((Ke1.flatten(), (self.i, self.j)))
        K1r = self.reduce_matrix(self.K1)
        u1r = self.reduce_vector(u1)

        a = 1.5 * Q0_norm**2 * (u1r.T @ K1r @ u1r)

        return a

    def get_koiter_ab(self, rhoE, lam_c, u0, u1, Q0_norm=None, norm=True):
        """
        u1: the critical buckling mode (without normalization)
        """
        if Q0_norm is None:
            Q0_norm = np.linalg.norm(u1)

        if norm:
            u1 = u1 / Q0_norm

        # if any of rhoE or u1 is complex, set the dtype to complex
        if np.iscomplexobj(rhoE) or np.iscomplexobj(u1):
            dt = complex
        else:
            dt = float

        ue0 = np.zeros((self.nelems, 8), dtype=dt)
        ue1 = np.zeros((self.nelems, 8), dtype=dt)

        ue0[:, ::2] = u0[2 * self.conn]
        ue0[:, 1::2] = u0[2 * self.conn + 1]

        ue1[:, ::2] = u1[2 * self.conn]
        ue1[:, 1::2] = u1[2 * self.conn + 1]

        # Compute the element stiffnesses
        if self.ptype_K == "simp":
            CK = np.outer(rhoE**self.p + self.rho0_K, self.C0)
        else:  # ramp
            CK = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_K, self.C0)

        CK = CK.reshape((self.nelems, 3, 3))

        Ke0 = np.zeros((self.nelems, 8, 8), dtype=dt)
        Ke1 = np.zeros((self.nelems, 8, 8), dtype=dt)
        Ke11 = np.zeros((self.nelems, 8, 8), dtype=dt)
        Ge0 = np.zeros((self.nelems, 8, 8), dtype=dt)
        Ge1 = np.zeros((self.nelems, 8, 8), dtype=dt)

        for i in range(4):
            detJ = self.detJ[:, i]
            Be = self.Be[:, :, :, i]
            Te = self.Te[:, :, :, :, i]

            Be1 = np.zeros((self.nelems, 3, 8), dtype=dt)
            populate_nonlinear_strain_and_Be(Be, ue1, Be1)

            CKBe = CK @ Be
            CKBe1 = CK @ Be1

            Ke0 += detJ.reshape(-1, 1, 1) * Be.transpose(0, 2, 1) @ CKBe
            Ke1 += detJ.reshape(-1, 1, 1) * Be.transpose(0, 2, 1) @ CKBe1
            Ke11 += detJ.reshape(-1, 1, 1) * Be1.transpose(0, 2, 1) @ CKBe1

            s0 = np.einsum("nik,nk -> ni", CKBe, ue0)
            G0e0 = np.einsum("n,ni,nijl -> njl", detJ, s0, Te)
            Ge0[:, 0::2, 0::2] += G0e0
            Ge0[:, 1::2, 1::2] += G0e0

            s1 = np.einsum("nik,nk -> ni", CKBe, ue1)
            G0e1 = np.einsum("n,ni,nijl -> njl", detJ, s1, Te)
            Ge1[:, 0::2, 0::2] += G0e1
            Ge1[:, 1::2, 1::2] += G0e1

        # Create sparse matrices directly
        K0 = sparse.csc_matrix((Ke0.flatten(), (self.i, self.j)))
        K1 = sparse.csc_matrix((Ke1.flatten(), (self.i, self.j)))
        K11 = sparse.csc_matrix((Ke11.flatten(), (self.i, self.j)))
        G0 = sparse.csc_matrix((Ge0.flatten(), (self.i, self.j)))
        G1 = sparse.csc_matrix((Ge1.flatten(), (self.i, self.j)))

        K0r = self.reduce_matrix(K0)
        K1r = self.reduce_matrix(K1)
        K11r = self.reduce_matrix(K11)
        G0r = self.reduce_matrix(G0)
        G1r = self.reduce_matrix(G1)
        u1r = self.reduce_vector(u1)

        # Formulate block matrix directly using sparse operations
        mu = -1.0 / lam_c
        Ar = G0r - mu * K0r
        Lr = (K0r @ u1r).reshape(-1, 1)  # SAME: L = (K1r @ u0r).reshape(-1, 1)
        rhs = mu * (G1r + 0.5 * K1r) @ u1r
        # nu = mu * u1r @ (0.5 * K1r + G1r) @ u1r * Q0_norm**2
        # rhs2 = -nu * K0r @ u1r
        # rhs = rhs1 + rhs2

        # u2r = sparse.linalg.spsolve(Ar, rhs)
        # u2 = self.full_vector(u2r)

        # check if Ar is singular
        # Ar = K0r + lam_c * G0r

        # Ar = K0r + lam_c * G0r
        # rhs = - (G1r + 0.5 * K1r) @ u1r

        # Construct the block matrix A in sparse format
        mat = sparse.bmat([[Ar, Lr], [Lr.T, [0]]], format="csc")

        # if np.linalg.matrix_rank(Ar.toarray()) < Ar.shape[0]:
        #     print("Ar is singular")
        # else:
        #     # check the condition number of Ar
        #     c = np.linalg.cond(Ar.toarray())
        #     print(f"Condition number of Ar: {c}")

        #     c = np.linalg.cond(mat.toarray())
        #     print(f"Condition number of mat: {c}")
        #     print("Ar is not singular")
        # exit()

        # Solve the linear system
        self.matfactor = linalg.factorized(mat)
        x = self.matfactor(np.hstack([rhs, 0]))

        # x = sparse.linalg.spsolve(mat, np.hstack([rhs, 0]))
        u2 = self.full_vector(x[:-1])
        u2r = self.reduce_vector(u2)
        self.nu = x[-1]

        a = 1.5 * Q0_norm**2 * (u1r.T @ K1r @ u1r)
        b = Q0_norm**2 * (
            u2r.T @ K1r @ u1r + 2 * u1r.T @ K1r @ u2r + 0.5 * u1r.T @ K11r @ u1r
        )

        return a, b, u1, u2, K1, K11, G1

    def get_lam_s(self, lam_c, a, xi=1e-3):
        # make sure a*xi < 0
        if a * xi > 0:
            xi = -xi

        x = np.abs(a * xi)
        lam_s = lam_c * (1 + 2 * x - 2 * np.sqrt(x**2 + x))

        return lam_s

    def get_ks_lams(self, a, xi=1e-3, ks_rho=160.0):
        lams = self.get_lam_s(self.BLF, a, xi)
        mu = 1 / lams
        c = max(mu)
        eta = np.exp(ks_rho * (mu - c))
        ks_min = c + np.log(np.sum(eta)) / ks_rho
        return ks_min

    def get_ks_lams_derivatives(self, a, dadx, xi=1e-3, ks_rho=160.0):
        if a * xi > 0:
            xi = -xi

        x = np.abs(a * xi)
        t0 = 1 + 2 * x - 2 * np.sqrt(x**2 + x)

        time0 = time.time()
        lams = self.BLF * t0
        mu = 1 / lams
        c = max(mu)
        eta = np.exp(ks_rho * (mu - c))
        eta = eta / np.sum(eta)

        dfdx = np.zeros(self.x.size)
        if self.deriv_type == "vector":
            for i in range(self.N):
                dKdx = self.get_stiffness_matrix_deriv(
                    self.rhoE, self.Q[:, i], self.Q[:, i]
                )
                dGdx = self.get_stress_stiffness_matrix_xderiv(
                    self.rhoE, self.u, self.Q[:, i], self.Q[:, i]
                )

                dGdu = self.get_stress_stiffness_matrix_uderiv(
                    self.rhoE, self.Q[:, i], self.Q[:, i]
                )
                dGdur = self.reduce_vector(dGdu)
                adjr = -self.Kfact(dGdur)
                adj = self.full_vector(adjr)

                dGdx += self.get_stiffness_matrix_deriv(self.rhoE, adj, self.u)
                # dfdrho -= eta[i] * (dGdx + mu[i] * dKdx)

                t1 = self.BLF[i] * (2 - (2 * x + 1) / np.sqrt(x**2 + x))

                dldx = self.BLF[i] * (dKdx + self.BLF[i] * dGdx)
                dfdx += eta[i] * (-1 / lams[i] ** 2) * dldx * t0
                dfdx -= eta[i] * (-1 / lams[i] ** 2) * dadx * xi * t1

        elif self.deriv_type == "tensor":

            eta_Q = (
                -eta[:, np.newaxis]
                / lams[:, np.newaxis] ** 2
                * self.BLF[:, np.newaxis]
                * self.Q.T
            ).T
            eta_BLF_Q = (
                -eta[:, np.newaxis]
                / lams[:, np.newaxis] ** 2
                * self.BLF[:, np.newaxis] ** 2
                * self.Q.T
            ).T

            dKdx = self.get_stiffness_matrix_deriv(self.rhoE, eta_Q, self.Q)

            dfds = self.intital_stress_stiffness_matrix_deriv(
                self.rhoE, self.Te, self.detJ, eta_BLF_Q, self.Q
            )
            dGdu = self.get_stress_stiffness_matrix_uderiv_tensor(dfds, self.Be)
            dGdur = self.reduce_vector(dGdu)
            adjr = -self.Kfact(dGdur)
            adj = self.full_vector(adjr)

            dGdx = self.get_stress_stiffness_matrix_xderiv_tensor(
                self.rhoE, self.u, dfds, self.Be
            )
            dGdx += self.get_stiffness_matrix_deriv(self.rhoE, adj, self.u)

            dfdx += (dGdx + dKdx) * t0

            for i in range(self.N):
                t1 = self.BLF[i] * (2 - (2 * x + 1) / np.sqrt(x**2 + x))
                dfdx += eta[i] / lams[i] ** 2 * dadx * xi * t1

        time1 = time.time()
        self.profile["total derivative time"] += time1 - time0

        return dfdx

    def get_flam_s(self, lam_c, a, xi=1e-3):
        return self.get_lam_s(lam_c, a, xi) / lam_c

    def get_dlamsdx(self, lam_c, a, dldx, dadx, xi=1e-3):
        if a * xi > 0:
            xi = -xi

        x = np.abs(a * xi)

        t0 = 1 + 2 * x - 2 * np.sqrt(x**2 + x)
        t1 = lam_c * (2 - (2 * x + 1) / np.sqrt(x**2 + x))

        return t0 * dldx - t1 * dadx * xi

    def get_dflamsdx(self, a, dadx, xi=1e-3):
        if a * xi > 0:
            xi = -xi

        x = np.abs(a * xi)
        tmp = 2 - (2 * x + 1) / np.sqrt(x**2 + x)
        return -tmp * dadx * xi

    def get_lams_b(self, lamc, kb, xib=1e-3):
        a = 1.0
        c = 3
        d = -1

        if kb < 0:
            b = -27 / 4 * kb * xib**2 - 3
        else:
            b = 27 / 4 * kb * xib**2 - 3

        p = (3 * a * c - b**2) / (3 * a**2)
        q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)
        D = (q / 2) ** 2 + (p / 3) ** 3

        print("D: ", D)

        if D > 0:
            u = np.cbrt(-q / 2 + np.sqrt(D))
            v = np.cbrt(-q / 2 - np.sqrt(D))
            t0 = u + v - b / (3 * a)
        else:
            theta = np.arccos(-q / 2 * np.sqrt(-27 / p**3))
            t0 = 2 * np.sqrt(-p / 3) * np.cos(theta / 3) - b / (3 * a)

        return lamc * t0

    def get_ks_lams_b(self, kb, xib=1e-3, ks_rho=160.0):
        lams = self.get_lams_b(self.BLF, kb, xib)
        mu = 1 / lams
        c = max(mu)
        eta = np.exp(ks_rho * (mu - c))
        ks_min = c + np.log(np.sum(eta)) / ks_rho
        return ks_min

    def get_dlams_b(self, lamc, kb, dldx, dkbdx, xib=1e-3):
        a = 1.0
        c = 3
        d = -1

        if kb < 0:
            b = -27 / 4 * kb * xib**2 - 3
        else:
            b = 27 / 4 * kb * xib**2 - 3

        p = (3 * a * c - b**2) / (3 * a**2)
        q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)
        D = (q / 2) ** 2 + (p / 3) ** 3

        if D > 0:
            u = np.cbrt(-q / 2 + np.sqrt(D))
            v = np.cbrt(-q / 2 - np.sqrt(D))
            t0 = u + v - b / (3 * a)
        else:
            w = -q / 2 * np.sqrt(-27 / p**3)
            theta = np.arccos(w)
            t0 = 2 * np.sqrt(-p / 3) * np.cos(theta / 3) - b / (3 * a)

        def df_dq():
            if D > 0:
                # Derivative of discriminant D with respect to q
                dD_dq = q / 2

                # Derivatives of a1 and a2 with respect to q
                da1_dq = -1 / 2 + (1 / 2) * (dD_dq / np.sqrt(D))
                da2_dq = -1 / 2 - (1 / 2) * (dD_dq / np.sqrt(D))

                # Applying the chain rule to cube roots
                b1 = (1 / (3 * u**2)) * da1_dq
                b2 = (1 / (3 * v**2)) * da2_dq

                dfdq = b1 + b2
            else:
                dtheta_da1 = -1 / np.sqrt(1 - w**2)
                da1_dq = -1 / 2 * np.sqrt(-27 / p**3)
                dtheta_dq = dtheta_da1 * da1_dq

                dxdtheta = 2 * np.sqrt(-p / 3) * (-np.sin(theta / 3) / 3)
                dfdq = dxdtheta * dtheta_dq

            return dfdq

        def df_dp():
            if D > 0:
                # Derivative of discriminant D with respect to p
                dD_dp = (p**2) / 9

                # Derivatives of a1 and a2 with respect to p
                da1_dp = (1 / 2) * (dD_dp / np.sqrt(D))
                da2_dp = -(1 / 2) * (dD_dp / np.sqrt(D))

                # Applying the chain rule to cube roots
                b1 = (1 / (3 * u**2)) * da1_dp
                b2 = (1 / (3 * v**2)) * da2_dp

                dfdp = b1 + b2
            else:
                dtheta_da1 = -1 / np.sqrt(1 - w**2)
                da1_dp = -q / 2 * (1 / 2) / np.sqrt(-27 / p**3) * 81 * p ** (-4)
                dtheta_dp = dtheta_da1 * da1_dp

                dxdtheta = 2 * np.sqrt(-p / 3) * (-np.sin(theta / 3) / 3)
                df_dp0 = 1 / np.sqrt(-p / 3) * (-1 / 3) * np.cos(theta / 3)
                dfdp = dxdtheta * dtheta_dp + df_dp0

            return dfdp

        dfdp = df_dp()
        dfdq = df_dq()

        dpdb = -2 * b / (3 * a**2)
        dqdb = (6 * b**2 - 9 * a * c) / (27 * a**3)

        dfdb = dfdp * dpdb + dfdq * dqdb - 1 / (3 * a)

        if kb < 0:
            dbdkb = -27 / 4 * xib**2
        else:
            dbdkb = 27 / 4 * xib**2

        t1 = dfdb * dbdkb

        return t0 * dldx + lamc * t1 * dkbdx

    def get_ks_lams_b_derivatives(self, kb, dkbdx, xib=1e-3, ks_rho=160.0):
        a = 1.0
        c = 3
        d = -1

        if kb < 0:
            b = -27 / 4 * kb * xib**2 - 3
        else:
            b = 27 / 4 * kb * xib**2 - 3

        p = (3 * a * c - b**2) / (3 * a**2)
        q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)
        D = (q / 2) ** 2 + (p / 3) ** 3

        if D > 0:
            u = np.cbrt(-q / 2 + np.sqrt(D))
            v = np.cbrt(-q / 2 - np.sqrt(D))
            t0 = u + v - b / (3 * a)
        else:
            w = -q / 2 * np.sqrt(-27 / p**3)
            theta = np.arccos(w)
            t0 = 2 * np.sqrt(-p / 3) * np.cos(theta / 3) - b / (3 * a)

        def df_dq():
            if D > 0:
                # Derivative of discriminant D with respect to q
                dD_dq = q / 2

                # Derivatives of a1 and a2 with respect to q
                da1_dq = -1 / 2 + (1 / 2) * (dD_dq / np.sqrt(D))
                da2_dq = -1 / 2 - (1 / 2) * (dD_dq / np.sqrt(D))

                # Applying the chain rule to cube roots
                b1 = (1 / (3 * u**2)) * da1_dq
                b2 = (1 / (3 * v**2)) * da2_dq

                dfdq = b1 + b2
            else:
                dtheta_da1 = -1 / np.sqrt(1 - w**2)
                da1_dq = -1 / 2 * np.sqrt(-27 / p**3)
                dtheta_dq = dtheta_da1 * da1_dq

                dxdtheta = 2 * np.sqrt(-p / 3) * (-np.sin(theta / 3) / 3)
                dfdq = dxdtheta * dtheta_dq

            return dfdq

        def df_dp():
            if D > 0:
                # Derivative of discriminant D with respect to p
                dD_dp = (p**2) / 9

                # Derivatives of a1 and a2 with respect to p
                da1_dp = (1 / 2) * (dD_dp / np.sqrt(D))
                da2_dp = -(1 / 2) * (dD_dp / np.sqrt(D))

                # Applying the chain rule to cube roots
                b1 = (1 / (3 * u**2)) * da1_dp
                b2 = (1 / (3 * v**2)) * da2_dp

                dfdp = b1 + b2
            else:
                dtheta_da1 = -1 / np.sqrt(1 - w**2)
                da1_dp = -q / 2 * (1 / 2) / np.sqrt(-27 / p**3) * 81 * p ** (-4)
                dtheta_dp = dtheta_da1 * da1_dp

                dxdtheta = 2 * np.sqrt(-p / 3) * (-np.sin(theta / 3) / 3)
                df_dp0 = 1 / np.sqrt(-p / 3) * (-1 / 3) * np.cos(theta / 3)
                dfdp = dxdtheta * dtheta_dp + df_dp0

            return dfdp

        dfdp = df_dp()
        dfdq = df_dq()

        dpdb = -2 * b / (3 * a**2)
        dqdb = (6 * b**2 - 9 * a * c) / (27 * a**3)

        dfdb = dfdp * dpdb + dfdq * dqdb - 1 / (3 * a)

        if kb < 0:
            dbdkb = -27 / 4 * xib**2
        else:
            dbdkb = 27 / 4 * xib**2

        t1 = dfdb * dbdkb

        time0 = time.time()

        lams = self.BLF * t0
        mu = 1 / lams
        c = max(mu)
        eta = np.exp(ks_rho * (mu - c))
        eta = eta / np.sum(eta)

        dfdx = np.zeros(self.x.size)

        eta_Q = (
            -eta[:, np.newaxis]
            / lams[:, np.newaxis] ** 2
            * self.BLF[:, np.newaxis]
            * self.Q.T
        ).T
        eta_BLF_Q = (
            -eta[:, np.newaxis]
            / lams[:, np.newaxis] ** 2
            * self.BLF[:, np.newaxis] ** 2
            * self.Q.T
        ).T

        dKdx = self.get_stiffness_matrix_deriv(self.rhoE, eta_Q, self.Q)

        dfds = self.intital_stress_stiffness_matrix_deriv(
            self.rhoE, self.Te, self.detJ, eta_BLF_Q, self.Q
        )
        dGdu = self.get_stress_stiffness_matrix_uderiv_tensor(dfds, self.Be)
        dGdur = self.reduce_vector(dGdu)
        adjr = -self.Kfact(dGdur)
        adj = self.full_vector(adjr)

        dGdx = self.get_stress_stiffness_matrix_xderiv_tensor(
            self.rhoE, self.u, dfds, self.Be
        )
        dGdx += self.get_stiffness_matrix_deriv(self.rhoE, adj, self.u)

        dfdx += (dGdx + dKdx) * t0

        for i in range(self.N):
            dfdx -= eta[i] / lams[i] ** 2 * self.BLF[i] * t1 * dkbdx

        time1 = time.time()
        self.profile["total derivative time"] += time1 - time0

        return dfdx

    def get_ks_lamc_b(self, kb, ks_rho=160.0):
        lams = self.BLF * kb
        mu = 1 / lams
        c = max(mu)
        eta = np.exp(ks_rho * (mu - c))
        ks_min = c + np.log(np.sum(eta)) / ks_rho
        return ks_min

    def get_ks_lamc_b_derivatives(self, kb, dkbdx, ks_rho=160.0):

        lams = self.BLF * kb
        mu = 1 / lams
        c = max(mu)
        eta = np.exp(ks_rho * (mu - c))
        eta = eta / np.sum(eta)

        dfdx = np.zeros(self.x.size)

        eta_Q = (
            -eta[:, np.newaxis]
            / lams[:, np.newaxis] ** 2
            * self.BLF[:, np.newaxis]
            * self.Q.T
        ).T
        eta_BLF_Q = (
            -eta[:, np.newaxis]
            / lams[:, np.newaxis] ** 2
            * self.BLF[:, np.newaxis] ** 2
            * self.Q.T
        ).T

        dKdx = self.get_stiffness_matrix_deriv(self.rhoE, eta_Q, self.Q)

        dfds = self.intital_stress_stiffness_matrix_deriv(
            self.rhoE, self.Te, self.detJ, eta_BLF_Q, self.Q
        )
        dGdu = self.get_stress_stiffness_matrix_uderiv_tensor(dfds, self.Be)
        dGdur = self.reduce_vector(dGdu)
        adjr = -self.Kfact(dGdur)
        adj = self.full_vector(adjr)

        dGdx = self.get_stress_stiffness_matrix_xderiv_tensor(
            self.rhoE, self.u, dfds, self.Be
        )
        dGdx += self.get_stiffness_matrix_deriv(self.rhoE, adj, self.u)

        dfdx += (dGdx + dKdx) * kb

        for i in range(self.N):
            dfdx -= eta[i] / lams[i] ** 2 * dkbdx * self.BLF[i]

        return dfdx

    def get_dldx(self, i=0):
        """
        d(lam)/dx = lam[0] * (dKdx + lam[0] * dGdx), where phi.T @ G @ phi = mu * I
        """
        u1 = self.Q[:, i]

        dKdx = self.get_stiffness_matrix_deriv(self.rhoE, u1, u1)
        dGdx = self.get_stress_stiffness_matrix_xuderiv(self.rhoE, self.u, u1, u1)

        return self.lam[i] * (dKdx + self.lam[i] * dGdx)

    def get_dQ0dx(self, dh=1e-4, p=None):
        """
        use central difference to compute du1/dx
        """
        # Use a random vector `p` if not provided
        if p is None:
            p = np.random.rand(self.x.size)
            p = p / np.linalg.norm(p)

        # Compute the perturbed design variables
        x1 = self.x + dh * p
        x2 = self.x - dh * p
        rho1 = self.fltr.apply(x1)
        rho2 = self.fltr.apply(x2)

        # prepare the stiffness matrices K1 and K2
        K1 = self.get_stiffness_matrix(rho1)
        K2 = self.get_stiffness_matrix(rho2)
        K1r = self.reduce_matrix(K1)
        K2r = self.reduce_matrix(K2)

        # solve for the solution path u01 and u02
        K1fact = linalg.factorized(K1r)
        K2fact = linalg.factorized(K2r)
        u01r = K1fact(self.fr)
        u02r = K2fact(self.fr)
        u01 = self.full_vector(u01r)
        u02 = self.full_vector(u02r)

        # prepare the stress stiffness matrices G1 and G2
        G1 = self.get_stress_stiffness_matrix(rho1, u01)
        G2 = self.get_stress_stiffness_matrix(rho2, u02)
        G1r = self.reduce_matrix(G1)
        G2r = self.reduce_matrix(G2)

        # solve the eigenvalue problem
        mu01, phi01 = sparse.linalg.eigsh(
            G1r, M=K1r, k=1, which="SM", sigma=self.lam[0]
        )
        mu02, phi02 = sparse.linalg.eigsh(
            G2r, M=K2r, k=1, which="SM", sigma=self.lam[0]
        )

        self.phi01 = self.full_vector(phi01[:, 0])
        self.phi02 = self.full_vector(phi02[:, 0])

        # check if the two eigenvalues are same direction, if not flip the sign
        if np.dot(self.phi01, self.phi02) < 0:
            self.phi01 *= -1

        du1dx = (self.phi01 - self.phi02) / (2 * dh)

        return du1dx

    def get_du1dx(self, dh=1e-4, p=None):
        """
        use central difference to compute du1/dx
        """
        self.get_dQ0dx(dh=dh, p=p)

        phi01 = self.phi01 / np.linalg.norm(self.phi01)
        phi02 = self.phi02 / np.linalg.norm(self.phi02)

        du1dx = (phi01 - phi02) / (2 * dh)

        return du1dx

    def get_exact_dQ0dx(self, Q0):
        """
        Compute the exact derivative of Q0 with respect to the design variables
        d(Q0)/dx = -1/2 (qi.T @ dKdx @ qi) * qi
        d(Q0)/dx += sum_j_(j!=i) (qj @ (dGdx - mu[i]*dKdx) / (mu_0 - mu_i)) * qj
        """
        # compute the full size eigenvalue problem
        mu, Qr = eigh(self.Gr.todense(), self.Kr.todense())
        # Q0 = self.full_vector(Qr[:, 0])
        # if np.dot(Q0, self.Q[:, 0]) < 0:
        # Q0 *= -1

        dKdx = self.get_stiffness_matrix_deriv(self.rhoE, Q0, Q0)

        t0 = time.time()
        dQ0dx = -0.5 * np.outer(Q0, dKdx)
        for i in range(1, Qr.shape[1]):
            qi = self.full_vector(Qr[:, i])

            dKdx = self.get_stiffness_matrix_deriv(self.rhoE, qi, Q0)
            dGdx = self.get_stress_stiffness_matrix_xuderiv(self.rhoE, self.u, qi, Q0)

            gx = (dGdx - mu[0] * dKdx) / (mu[0] - mu[i])
            dQ0dx += np.outer(qi, gx)

            _print_progress(i, Qr.shape[1], t0, "dQ0/dx")

        return dQ0dx

    def get_dG0dx_dK0dx_du0dx_cs(self, x, K, G, u0, Q0, dh=1e-30):
        """
        Compute the derivative of the stiffness matrix times the vectors u
        """
        # Preallocate memory for results before parallelization
        ndv = x.size
        dG0dx = np.zeros((2 * self.nnodes, ndv), dtype=x.dtype)
        dK0dx = np.zeros((2 * self.nnodes, ndv), dtype=x.dtype)
        du0dx = np.zeros((2 * self.nnodes, ndv), dtype=x.dtype)

        def iterator(ii):
            # Copy x and adjust the current index
            x_cs = x.copy().astype(complex)
            x_cs[ii] += dh * 1j

            # Apply filter and compute stiffness matrix
            rho = self.fltr.apply(x_cs)
            K1 = self.get_stiffness_matrix(rho)

            # Reduce and solve the system for the current perturbation
            K1r = self.reduce_matrix(K1)
            u1r = sparse.linalg.spsolve(K1r, self.fr)

            # Remaining operations
            u01 = self.full_vector(u1r)
            G1 = self.get_stress_stiffness_matrix(rho, u01)

            # Compute derivatives
            dG0dx_n = (G1 - G).imag @ Q0 / dh
            dK0dx_n = (K1 - K).imag @ Q0 / dh
            du0dx_n = (u01 - u0).imag / dh

            _print_progress(ii, self.nnodes, t0, "dQ0/dx")

            return dG0dx_n, dK0dx_n, du0dx_n

        t0 = time.time()

        # Parallelize the loop
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(iterator)(ii) for ii in range(ndv)
        )

        # Unpack results efficiently
        for n, (dG0dx_n, dK0dx_n, du0dx_n) in enumerate(results):
            dG0dx[:, n] = dG0dx_n
            dK0dx[:, n] = dK0dx_n
            du0dx[:, n] = du0dx_n

        return dK0dx, dG0dx, du0dx

    @timeit(print_time=True)
    def get_dQ0dx_dl0dx(self, x, K0, G0, dK0dx, dG0dx, l0, Q0):
        """
        Compute:
            eigenvector derivatives dQ0/dx
            eigenvalue derivative dl0/dx

        An efficient algebraic method for the computation of natural frequency and mode shape sensitivities—Part I. Distinct natural frequencies - Lee, In-Won (1997)
        https://www.sciencedirect.com/science/article/pii/S0045794996002064
        """
        mu = -1.0 / l0

        # Apply filter and calculate rhoE only once
        rhoE = np.mean(self.fltr.apply(x)[self.conn[:, :4]], axis=1)

        # Construct the block matrix A in sparse format
        A = G0 - mu * K0
        L = (K0 @ Q0).reshape(-1, 1)
        a = sparse.csc_matrix(([0], ([0], [0])), shape=(1, 1))

        Ar = self.reduce_matrix(A)
        Lr = self.reduce_vector(L)
        mat = sparse.bmat([[Ar, -Lr], [-Lr.T, a]], format="csc")

        # Construct the right-hand side
        dKdx = self.get_stiffness_matrix_deriv(rhoE, Q0, Q0)

        # dG0dx, dK0dx, du0dx = self.get_dG0dx_dK0dx_du0dx_cs(x, K0, G0, u0, Q0)
        rhs1 = dG0dx - mu * dK0dx

        rhs1r = np.array([self.reduce_vector(rhs1[:, i]) for i in range(self.nnodes)]).T
        rhs = np.vstack([-rhs1r, 0.5 * dKdx.T])

        # Solve the linear system
        x = sparse.linalg.spsolve(mat, rhs)

        # Reconstruct the eigenvector derivatives dQ0/dx and eigenvalue derivative dl0/dx
        dQ0dx = np.array([self.full_vector(x[:-1, i]) for i in range(self.nnodes)]).T
        dl0dx = l0**2 * x[-1]

        return dQ0dx, dl0dx

    def get_exact_du1dx(self, Q0):
        """
        Compute the exact derivative of Q0 with respect to the design variables
        d(u1)/dx = 1/u1_norm * (d(Q0)/dx - 1/u1_norm**3 * Q0 * (Q0^T @ d(Q0)/dx))
        """
        du1dx = self.get_exact_dQ0dx()
        u1_norm = np.linalg.norm(Q0)
        du1dx_normalized = du1dx / u1_norm - np.outer(Q0, 1 / u1_norm**3 * Q0.T @ du1dx)

        return du1dx_normalized

    def get_dK1du(self, rhoE, u, v):
        """
        Compute: u^T @ d(K1)/du @ v   <-- which is a vector
        """

        # Get the element-wise solution variables
        ue = np.zeros((self.nelems, 8), dtype=rhoE.dtype)
        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        # Compute the element stiffnesses
        if self.ptype_K == "simp":
            CK = np.outer(rhoE**self.p + self.rho0_K, self.C0)
        else:  # ramp
            CK = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_K, self.C0)

        CK = CK.reshape((self.nelems, 3, 3))

        Ge1 = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)

        for i in range(4):
            detJ = self.detJ[:, i]
            Be = self.Be[:, :, :, i]
            Te = self.Te[:, :, :, :, i]

            CKBe = CK @ Be
            s1 = np.einsum("nik,nk -> ni", CKBe, ue)
            G0e1 = detJ.reshape(-1, 1, 1) * np.einsum("ni,nijl -> njl", s1, Te)
            Ge1[:, 0::2, 0::2] += G0e1
            Ge1[:, 1::2, 1::2] += G0e1

        dKdu = sparse.csc_matrix((Ge1.flatten(), (self.i, self.j)))

        return dKdu @ v

    @timeit(print_time=True)
    def get_dK1dQ0_cs(self, rhoE, K1, Q, u, v, dh=1e-20):

        dK1dQ0 = np.zeros(2 * self.nnodes)

        for i in range(2 * self.nnodes):
            Q1 = Q.copy().astype(complex)
            Q1[i] += 1j * dh

            self.get_koiter_a(rhoE, Q1)
            dK1dQ0[i] = (u.T @ (self.K1 - K1) @ v).imag / dh

        return dK1dQ0

    @timeit(print_time=True)
    def get_dK11du1_cs(self, rhoE, K11, Q, u, v, dh=1e-30):

        dK1du1 = np.zeros(2 * self.nnodes)
        Q_norm = np.linalg.norm(Q)
        u1 = Q / Q_norm

        for i in range(2 * self.nnodes):
            u1_1 = u1.copy().astype(complex)
            u1_1[i] += 1j * dh

            K11_1 = self.get_K11(rhoE, u1_1)
            dK1du1[i] = (u.T @ (K11_1 - K11) @ v).imag / dh

        return dK1du1

    @timeit(print_time=True)
    def get_dK1dQ0_cd(self, rhoE, Q, u, v, dh=1e-4):

        dK1du1 = np.zeros(2 * self.nnodes)

        Q_norm = np.linalg.norm(Q)
        u1 = Q / Q_norm

        # Compute the element stiffnesses
        if self.ptype_K == "simp":
            C = np.outer(rhoE**self.p + self.rho0_K, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_K, self.C0)

        C = C.reshape((self.nelems, 3, 3))

        for i in range(2 * self.nnodes):
            u1_1 = u1.copy()
            u1_2 = u1.copy()

            u1_1[i] += dh
            u1_2[i] -= dh

            # u1_1_norm = np.linalg.norm(u1_1)
            # u1_2_norm = np.linalg.norm(u1_2)

            # u1_1 = u1_1 / np.linalg.norm(u1_1)
            # u1_2 = u1_2 / np.linalg.norm(u1_2)

            ue1 = np.zeros((self.nelems, 8))
            ue2 = np.zeros((self.nelems, 8))
            ue1[:, ::2] = u1_1[2 * self.conn]
            ue1[:, 1::2] = u1_1[2 * self.conn + 1]
            ue2[:, ::2] = u1_2[2 * self.conn]
            ue2[:, 1::2] = u1_2[2 * self.conn + 1]

            dKe1 = np.zeros((self.nelems, 8, 8))

            for j in range(4):
                detJ = self.detJ[:, j]
                Be = self.Be[:, :, :, j]

                Be1 = np.zeros((self.nelems, 3, 8))
                Be2 = np.zeros((self.nelems, 3, 8))
                populate_nonlinear_strain_and_Be(Be, ue1, Be1)
                populate_nonlinear_strain_and_Be(Be, ue2, Be2)

                dCBe = C @ (Be1 - Be2)
                dKe1 += detJ.reshape(-1, 1, 1) * Be.transpose(0, 2, 1) @ dCBe

            dK1 = sparse.csc_matrix((dKe1.flatten(), (self.i, self.j)))
            # dK1du1[i] = (u.T @ dK1 @ v) / (dh / u1_1_norm + dh / u1_2_norm)
            dK1du1[i] = (u.T @ dK1 @ v) / (2 * dh)

        return dK1du1 / Q_norm

        # for i in range(2 * self.nnodes):
        #     Q1 = Q.copy()
        #     Q2 = Q.copy()
        #     Q1[i] += dh * np.linalg.norm(Q)
        #     Q2[i] -= dh * np.linalg.norm(Q)

        #     self.get_koiter_a(rhoE, Q1)
        #     K1_1 = self.K1
        #     self.get_koiter_a(rhoE, Q2)
        #     K1_2 = self.K1
        #     dK1du1[i] = (u.T @ (K1_1 - K1_2) @ v) / (2 * dh)

        # return dK1du1 / np.linalg.norm(Q)

    @timeit(print_time=True)
    def get_dK11du1_cd(self, rhoE, Q, u, v, dh=1e-4):

        dK11du1 = np.zeros(2 * self.nnodes)

        Q_norm = np.linalg.norm(Q)
        u1 = Q / Q_norm

        # Compute the element stiffnesses
        if self.ptype_K == "simp":
            C = np.outer(rhoE**self.p + self.rho0_K, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_K, self.C0)

        C = C.reshape((self.nelems, 3, 3))

        for i in range(2 * self.nnodes):
            u1_1 = u1.copy()
            u1_2 = u1.copy()

            u1_1[i] += dh
            u1_2[i] -= dh

            ue1 = np.zeros((self.nelems, 8))
            ue2 = np.zeros((self.nelems, 8))
            ue1[:, ::2] = u1_1[2 * self.conn]
            ue1[:, 1::2] = u1_1[2 * self.conn + 1]
            ue2[:, ::2] = u1_2[2 * self.conn]
            ue2[:, 1::2] = u1_2[2 * self.conn + 1]

            dKe11 = np.zeros((self.nelems, 8, 8))

            for j in range(4):
                detJ = self.detJ[:, j]
                Be = self.Be[:, :, :, j]

                Be1 = np.zeros((self.nelems, 3, 8))
                Be2 = np.zeros((self.nelems, 3, 8))
                populate_nonlinear_strain_and_Be(Be, ue1, Be1)
                populate_nonlinear_strain_and_Be(Be, ue2, Be2)

                BeCBe1 = Be1.transpose(0, 2, 1) @ C @ Be1
                BeCBe2 = Be2.transpose(0, 2, 1) @ C @ Be2
                dKe11 += detJ.reshape(-1, 1, 1) * (BeCBe1 - BeCBe2)

            dK11 = sparse.csc_matrix((dKe11.flatten(), (self.i, self.j)))
            dK11du1[i] = (u.T @ dK11 @ v) / (2 * dh)

            # K11_1 = self.get_K11(rhoE, u1_1)
            # K11_2 = self.get_K11(rhoE, u1_2)
            # dK11du1[i] = (u.T @ (K11_1 - K11_2) @ v) / (2 * dh)

        return dK11du1

    def get_dadQ0_cd(self, rhoE, Q, dh=1e-4):

        dadQ0 = np.zeros(2 * self.nnodes)

        for i in range(2 * self.nnodes):
            Q_1 = Q.copy()
            Q_2 = Q.copy()
            Q_1[i] += dh
            Q_2[i] -= dh

            a1 = self.get_koiter_a(rhoE, Q_1)
            a2 = self.get_koiter_a(rhoE, Q_2)

            dadQ0[i] = (a1 - a2) / (2 * dh)

        return dadQ0

    @timeit(print_time=True)
    def get_dbdu1_cs(self, rhoE, lam_c, Q0, dh=1e-30):

        Q0_norm = np.linalg.norm(Q0)
        u1 = Q0 / Q0_norm

        dbdu1 = np.zeros(2 * self.nnodes)

        for i in range(2 * self.nnodes):
            u1_1 = u1.copy().astype(complex)
            u1_1[i] += 1j * dh

            K1 = self.get_K1(rhoE, u1_1)
            K11 = self.get_K11(rhoE, u1_1)
            b1 = Q0_norm**2 * (
                self.u2 @ K1 @ u1_1 + 0.5 * u1_1 @ K11 @ u1_1 + 2 * u1_1 @ K1 @ self.u2
            )

            dbdu1[i] = (b1 - self.b).imag / dh

        return dbdu1

    def get_dbdu1_cd(self, rhoE, lam_c, Q0, dh=1e-6):

        Q0_norm = np.linalg.norm(Q0)
        u1 = Q0 / Q0_norm

        dbdu1 = np.zeros(2 * self.nnodes)

        for i in range(2 * self.nnodes):
            u1_1 = u1.copy()
            u1_2 = u1.copy()
            u1_1[i] += dh
            u1_2[i] -= dh

            K1_1 = self.get_K1(rhoE, u1_1)
            K11_1 = self.get_K11(rhoE, u1_1)
            b1 = Q0_norm**2 * (
                self.u2 @ K1_1 @ u1_1
                + 0.5 * u1_1 @ K11_1 @ u1_1
                + 2 * u1_1 @ K1_1 @ self.u2
            )

            K1_2 = self.get_K1(rhoE, u1_2)
            K11_2 = self.get_K11(rhoE, u1_2)
            b2 = Q0_norm**2 * (
                self.u2 @ K1_2 @ u1_2
                + 0.5 * u1_2 @ K11_2 @ u1_2
                + 2 * u1_2 @ K1_2 @ self.u2
            )

            dbdu1[i] = (b1 - b2) / (2 * dh)

        return dbdu1

    def get_dbdu2_cs(self, rhoE, lam_c, Q0, dh=1e-30):

        Q0_norm = np.linalg.norm(Q0)
        u1 = Q0 / Q0_norm

        dbdu2 = np.zeros(2 * self.nnodes)

        for i in range(2 * self.nnodes):
            u2_1 = self.u2.copy().astype(complex)
            u2_1[i] += 1j * dh

            b1 = Q0_norm**2 * (
                u2_1 @ self.K1 @ self.u1
                + 0.5 * self.u1 @ self.K11 @ self.u1
                + 2 * self.u1 @ self.K1 @ u2_1
            )

            dbdu2[i] = (b1 - self.b).imag / dh

        return dbdu2

    def get_papx_cs(self, x, a, Q, dh=1e-30):

        papx = np.zeros(self.nnodes)

        for i in range(self.nnodes):
            x1 = x.copy().astype(complex)
            x1[i] += 1j * dh

            rhoE1 = np.mean(self.fltr.apply(x1)[self.conn[:, :4]], axis=1)
            a1 = self.get_koiter_a(rhoE1, Q)
            papx[i] = (a1 - a).imag / dh

        return papx

    @timeit(print_time=True)
    def get_pbpx_cs(self, x, lam_c, Q, dh=1e-30):

        pbpx = np.zeros(self.nnodes)

        for i in range(self.nnodes):
            x1 = x.copy().astype(complex)
            x1[i] += 1j * dh

            rhoE1 = np.mean(self.fltr.apply(x1)[self.conn[:, :4]], axis=1)
            # b1 = self.get_koiter_ab(rhoE1, lam_c, Q)[1]--> which is wrong u2 is changed
            _, _, _, _, K1, K11, G1 = self.get_koiter_ab(rhoE1, lam_c, Q)
            b1 = np.linalg.norm(Q) ** 2 * (
                self.u2 @ K1 @ self.u1
                + 0.5 * self.u1 @ K11 @ self.u1
                + 2 * self.u1 @ K1 @ self.u2
            )
            pbpx[i] = (b1 - self.b).imag / dh

        return pbpx

    @timeit(print_time=True)
    def get_pbpx_cd(self, x, lam_c, Q, dh=1e-4):

        pbpx = np.zeros(self.nnodes)

        for i in range(self.nnodes):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] += dh
            x2[i] -= dh

            rhoE1 = np.mean(self.fltr.apply(x1)[self.conn[:, :4]], axis=1)
            rhoE2 = np.mean(self.fltr.apply(x2)[self.conn[:, :4]], axis=1)
            b1 = self.get_koiter_ab(rhoE1, lam_c, self.u, Q)[1]
            b2 = self.get_koiter_ab(rhoE2, lam_c, self.u, Q)[1]

            pbpx[i] = (b1 - b2) / (2 * dh)

        return pbpx

    def get_dK11dx_cs(self, x, rhoE, u1, u, v, dh=1e-30):

        dK11dx = np.zeros(x.size)

        for i in range(x.size):
            x1 = x.copy().astype(complex)
            x1[i] += 1j * dh

            rhoE1 = np.mean(self.fltr.apply(x1)[self.conn[:, :4]], axis=1)
            K11_1 = self.get_K11(rhoE1, u1)
            dK11dx[i] = (u @ (K11_1 - self.K11) @ v).imag / dh

        return dK11dx

    def get_dG1dx_cs(self, x, rhoE, u1, u, v, dh=1e-30):

        dG1dx = np.zeros(x.size)

        for i in range(x.size):
            x1 = x.copy().astype(complex)
            x1[i] += 1j * dh

            rhoE1 = np.mean(self.fltr.apply(x1)[self.conn[:, :4]], axis=1)
            G1_1 = self.get_G1(rhoE1, u1)
            dG1dx[i] = (u @ (G1_1 - self.G1) @ v).imag / dh

        return dG1dx

    def get_dG1dx_cd(self, x, rhoE, u1, u, v, dh=1e-4):

        dG1dx = np.zeros(x.size)

        for i in range(x.size):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] += dh
            x2[i] -= dh

            rhoE1 = np.mean(self.fltr.apply(x1)[self.conn[:, :4]], axis=1)
            rhoE2 = np.mean(self.fltr.apply(x2)[self.conn[:, :4]], axis=1)
            G1_1 = self.get_G1(rhoE1, u1)
            G1_2 = self.get_G1(rhoE2, u1)
            dG1dx[i] = (u @ (G1_1 - G1_2) @ v) / (2 * dh)

        return dG1dx

    def get_dadx_dbdx_cd(self, x, du0dx, dl0dx, dQ0dx, u0, l0, Q0, dh=1e-4):
        """
        Compute the derivative of the stiffness matrix times the vectors u
        """
        t0 = time.time()

        conn_indices = self.conn[:, :4]
        Q0_norm = np.linalg.norm(Q0)
        u1 = Q0 / Q0_norm

        dadx = np.zeros(x.size)
        dbdx = np.zeros(x.size)

        for i in range(self.nnodes):
            x1 = x.copy()
            x2 = x.copy()

            x1[i] += dh
            x2[i] -= dh

            rhoE1 = np.mean(self.fltr.apply(x1)[conn_indices], axis=1)
            rhoE2 = np.mean(self.fltr.apply(x2)[conn_indices], axis=1)

            l0_1 = l0 + dl0dx[i] * dh
            l0_2 = l0 - dl0dx[i] * dh

            u0_1 = u0 + du0dx[:, i] * dh
            u0_2 = u0 - du0dx[:, i] * dh

            u1_1 = u1 + dQ0dx[:, i] * dh / Q0_norm
            u1_2 = u1 - dQ0dx[:, i] * dh / Q0_norm

            a1, b1 = self.get_koiter_ab(rhoE1, l0_1, u0_1, u1_1, Q0_norm, False)[0:2]
            a2, b2 = self.get_koiter_ab(rhoE2, l0_2, u0_2, u1_2, Q0_norm, False)[0:2]

            dadx[i] = (a1 - a2) / (2 * dh)
            dbdx[i] = (b1 - b2) / (2 * dh)

            _print_progress(i, x.size, t0, "da/dx")

        return dadx, dbdx

    def get_dadx_dbdx_cs(self, x, du0dx, dl0dx, dQ0dx, u0, l0, Q0, dh=1e-30):

        dadx = np.zeros(x.size)
        dbdx = np.zeros(x.size)

        t0 = time.time()

        conn_indices = self.conn[:, :4]
        Q0_norm = np.linalg.norm(Q0)
        u1 = Q0 / Q0_norm

        for i in range(self.nnodes):
            x1 = x.copy().astype(complex)
            x1[i] += 1j * dh

            rhoE1 = np.mean(self.fltr.apply(x1)[conn_indices], axis=1)
            l0_1 = l0 + dl0dx[i] * 1j * dh
            Q0_1 = Q0 + dQ0dx[:, i] * 1j * dh
            u0_1 = u0 + du0dx[:, i] * 1j * dh

            a1, b1 = self.get_koiter_ab(rhoE1, l0_1, u0_1, Q0_1, Q0_norm)[0:2]

            dadx[i] = (a1 - self.a).imag / dh
            dbdx[i] = (b1 - self.b).imag / dh

            _print_progress(i, x.size, t0, "da/dx")

        return dadx, dbdx

        # dadx = np.zeros(x.size)

        # def iterator(n):
        #     x1 = x.copy()
        #     x2 = x.copy()

        #     x1[n] += dh
        #     x2[n] -= dh

        #     dxi = 2 * dh
        #     dQ0 = dQ0dx[:, n] * dxi

        #     # Apply filtering and compute average rhoE for both perturbed vectors
        #     rhoE1 = np.mean(self.fltr.apply(x1)[conn_indices], axis=1)
        #     rhoE2 = np.mean(self.fltr.apply(x2)[conn_indices], axis=1)

        #     u1 = Q0 / np.linalg.norm(Q0)
        #     u1_1 = u1 + dQ0dx[:, n] * dh / np.linalg.norm(Q0)
        #     u1_2 = u1 - dQ0dx[:, n] * dh / np.linalg.norm(Q0)

        #     # Q0_1 = Q0 + dQ0dx[:, n] * dh
        #     # Q0_2 = Q0 - dQ0dx[:, n] * dh

        #     # # Normalize Q1 and Q2
        #     # u1_1_norm = np.linalg.norm(Q0_1)
        #     # u1_2_norm = np.linalg.norm(Q0_2)
        #     # u1_1 = Q0_1 / u1_1_norm
        #     # u1_2 = Q0_2 / u1_2_norm

        #     ue1_1 = np.zeros((self.nelems, 8), dtype=rhoE1.dtype)
        #     ue1_2 = np.zeros((self.nelems, 8), dtype=rhoE2.dtype)
        #     ue1_1[:, ::2] = u1_1[2 * self.conn]
        #     ue1_1[:, 1::2] = u1_1[2 * self.conn + 1]
        #     ue1_2[:, ::2] = u1_2[2 * self.conn]
        #     ue1_2[:, 1::2] = u1_2[2 * self.conn + 1]

        #     # Compute the element stiffnesses
        #     if self.ptype_K == "simp":
        #         C1 = np.outer(rhoE1**self.p + self.rho0_K, self.C0)
        #         C2 = np.outer(rhoE2**self.p + self.rho0_K, self.C0)
        #     else:  # ramp
        #         C1 = np.outer(
        #             rhoE1 / (1.0 + self.q * (1.0 - rhoE1)) + self.rho0_K, self.C0
        #         )
        #         C2 = np.outer(
        #             rhoE2 / (1.0 + self.q * (1.0 - rhoE2)) + self.rho0_K, self.C0
        #         )

        #     C1 = C1.reshape((self.nelems, 3, 3))
        #     C2 = C2.reshape((self.nelems, 3, 3))
        #     dKe1 = np.zeros((self.nelems, 8, 8), dtype=rhoE1.dtype)

        #     for i in range(4):
        #         detJ = self.detJ[:, i]
        #         Be = self.Be[:, :, :, i]

        #         Be1_1 = np.zeros((self.nelems, 3, 8), dtype=rhoE1.dtype)
        #         Be1_2 = np.zeros((self.nelems, 3, 8), dtype=rhoE2.dtype)

        #         populate_nonlinear_strain_and_Be(Be, ue1_1, Be1_1)
        #         populate_nonlinear_strain_and_Be(Be, ue1_2, Be1_2)

        #         dCKBe1 = C1 @ Be1_1 - C2 @ Be1_2
        #         dKe1 += detJ.reshape(-1, 1, 1) * Be.transpose(0, 2, 1) @ dCKBe1

        #     dK1 = sparse.csc_matrix((dKe1.flatten(), (self.i, self.j)))

        #     dadxi = 1.5 * (Q0.T @ dK1 + dQ0.T @ (K1 + K1.T)) @ Q0 / dxi

        #     _print_progress(n, self.nnodes, t0, "da/dx")

        #     return dadxi

        # results = Parallel(n_jobs=n_jobs, prefer="threads")(
        #     delayed(iterator)(n) for n in range(x.size)
        # )

        # for n, dadxi in enumerate(results):
        #     dadx[n] = dadxi

        return dadx

    def get_dadx(self, rhoE, l0, Q0):
        Q0_norm = np.linalg.norm(Q0)
        Q0r = self.reduce_vector(Q0)
        K1r = self.reduce_matrix(self.K1)

        # Ar = self.Gr + self.Kr / l0
        # Lr = (self.Kr @ Q0r).reshape(-1, 1)
        # mat = sparse.bmat([[Ar, -Lr], [-Lr.T, [0]]], format="csc")

        # dK1dQ0_cd = self.get_dK1dQ0_cd(rhoE, Q0, Q0, Q0)
        # dK1dQ0_cd = self.reduce_vector(dK1dQ0_cd)

        # dK1dQ0_cs = self.get_dK1dQ0_cs(rhoE, K1, Q0, Q0, Q0)
        # dK1dQ0_cs = self.reduce_vector(dK1dQ0_cs)

        Qb = Q0r @ (K1r.T + 2 * K1r)
        rhs = np.hstack([Qb, 0])
        psi = self.matfactor(-1.5 * rhs)
        # psi = sparse.linalg.spsolve(mat, -1.5 * rhs)

        psi0 = self.full_vector(psi[:-1])
        psi1 = -psi[-1] / Q0_norm

        dGdx1 = self.get_stress_stiffness_matrix_xuderiv(rhoE, self.u, psi0, Q0)
        dKdx1 = self.get_stiffness_matrix_deriv(rhoE, psi0, Q0)
        dKdx2 = self.get_stiffness_matrix_deriv(rhoE, Q0, Q0)
        dK1dx = self.get_dK1dx(rhoE, Q0, Q0, Q0)

        dadx = (dGdx1 + dKdx1 / l0) - 0.5 * psi1 * dKdx2 + 1.5 * dK1dx

        return dadx

    def get_dbdx(self, rhoE, l0, Q0, dl0dx=None, du0dx=None, dQ0dx=None, dh=1e-30):
        u0 = self.u
        u1 = self.u1
        u2 = self.u2
        u1r = self.reduce_vector(u1)
        u2r = self.reduce_vector(u2)

        K0r = self.reduce_matrix(self.K0)
        # G0r = self.reduce_matrix(self.G0)
        K1r = self.reduce_matrix(self.K1)
        G1r = self.reduce_matrix(self.G1)
        K11r = self.reduce_matrix(self.K11)

        mu = -1.0 / l0
        Q0_norm = np.linalg.norm(Q0)
        nu = mu * u1r @ (0.5 * K1r + G1r) @ u1r * Q0_norm**2

        pbpu1 = 2 * Q0_norm**2 * (u2r @ (K1r + K1r.T + G1r) + u1r @ K11r)
        pbpu2 = Q0_norm**2 * u1r @ (2 * K1r + K1r.T)

        # solve for governing equations R2
        # Ar = G0r - mu * K0r
        # Lr = (K0r @ u1r).reshape(-1, 1)
        # mat = sparse.bmat([[Ar, Lr], [Lr.T, [0]]], format="csc")
        rhs = np.hstack([pbpu2, 0])

        psi = self.matfactor(-rhs)
        # psi = sparse.linalg.spsolve(mat.T, -rhs)
        pu2r = psi[:-1]
        pu2 = self.full_vector(pu2r)
        pnu = psi[-1]

        # solve for governing equations R1
        # mat = sparse.bmat(
        #     [[Ar * Q0_norm, -Lr * Q0_norm], [-(Q0_norm**2) * Lr.T, [0]]], format="csc"
        # )
        pR21pu1 = nu * K0r - mu * (K1r + K1r.T + G1r)
        pR22pu1 = u2r @ K0r
        pfpu1 = pbpu1 + pu2r @ pR21pu1 + pnu * pR22pu1
        pfpmu = -pu2r @ K0r @ u2r - pu2r @ (0.5 * K1r + G1r) @ u1r
        rhs = np.hstack([pfpu1, -pfpmu])

        # psi = sparse.linalg.spsolve(mat.T, -rhs)
        psi = self.matfactor(-rhs)
        pu1r = psi[:-1] / Q0_norm
        pu1 = self.full_vector(pu1r)
        pmu = -psi[-1] / Q0_norm**2

        # compute the partial derivative pb/px
        u2_dK1_u1 = self.get_dK1dx(rhoE, Q0, u2, u1)
        u1_dK1_u2 = self.get_dK1dx(rhoE, Q0, u1, u2)
        u1_dK11_u1 = self.get_dK11dx(rhoE, Q0, u1, u1)
        pbpx = Q0_norm**2 * (u2_dK1_u1 + 0.5 * u1_dK11_u1 + 2 * u1_dK1_u2)

        # compute pR21 / px
        pu2_dG0dx_u2 = self.get_stress_stiffness_matrix_xuderiv(rhoE, u0, pu2, u2)
        pu2_dK0dx_u2 = self.get_stiffness_matrix_deriv(rhoE, pu2, u2)
        pu2_dK0dx_u1 = self.get_stiffness_matrix_deriv(rhoE, pu2, u1)
        pu2_dK1_u1 = self.get_dK1dx(rhoE, Q0, pu2, u1)
        pu2_dG1_u1 = self.get_stress_stiffness_matrix_xderiv(rhoE, u1, pu2, u1)
        pu2_pR21px = (
            pu2_dG0dx_u2
            - mu * pu2_dK0dx_u2
            + nu * pu2_dK0dx_u1
            - mu * (0.5 * pu2_dK1_u1 + pu2_dG1_u1)
        )

        # compute pR22 / px
        u1_dK0dx_u2 = self.get_stiffness_matrix_deriv(rhoE, u1, u2)
        pR22px = u1_dK0dx_u2

        # compute the partial derivative pR11/px
        pu1_dG0dx_Q0 = self.get_stress_stiffness_matrix_xuderiv(rhoE, u0, pu1, Q0)
        pu1_dK0dx_Q0 = self.get_stiffness_matrix_deriv(rhoE, pu1, Q0)
        pu1_pR11px = pu1_dG0dx_Q0 - mu * pu1_dK0dx_Q0

        # compute the partial derivative pR12/px
        Q0_dK0dx_Q0 = self.get_stiffness_matrix_deriv(rhoE, Q0, Q0)
        pR12px = -0.5 * Q0_dK0dx_Q0

        # compute the total derivative db/dx
        dbdx = pbpx + pu2_pR21px + pnu * pR22px + pu1_pR11px + pmu * pR12px

        return dbdx

    @timeit(print_time=True)
    def get_dal0dx(self, l0, a, dl0dx, dadx):
        dal0dx = dadx * l0 + a * dl0dx
        return dal0dx

    @timeit(print_time=True)
    def get_derivative_cd(
        self,
        x,
        K0,
        K1,
        K11,
        G0,
        G1,
        du0dx,
        dQ0dx,
        dl0dx,
        l0,
        u0,
        Q0,
        u2,
        u,
        v,
        dh=1e-4,
        Q0_norm=None,
    ):
        """
        Compute the derivative of the stiffness matrix times the vectors u
        """
        u1_norm = Q0_norm if Q0_norm is not None else np.linalg.norm(Q0)
        u1 = Q0 / u1_norm
        u1r = self.reduce_vector(u1)
        dK1dx = np.zeros(self.nnodes)
        dadx = np.zeros(self.nnodes)
        dbdx = np.zeros(self.nnodes)
        dbdx_temp = np.zeros(self.nnodes)
        rhs_mat = np.zeros((u1r.size + 1, self.nnodes))
        # dQdx = np.zeros((2 * self.nnodes, self.nnodes))

        # Precompute detJ * Be terms outside the loop to avoid redundancy
        detJBe = np.zeros((self.nelems, 8, 3, 4))
        for i in range(4):
            Be = self.Be[:, :, :, i]
            detJ = self.detJ[:, i]
            detJBe[:, :, :, i] = detJ.reshape(-1, 1, 1) * Be.transpose(0, 2, 1)

        conn_indices = self.conn[:, :4]

        t0 = time.time()
        for n in range(x.size):
            x1 = x.copy()
            x2 = x.copy()

            x1[n] += dh
            x2[n] -= dh

            # dxi = 2 * dh
            # dQ0 = dQ0dx[:, n] * dxi

            # Apply filtering and compute average rhoE for both perturbed vectors
            rhoE1 = np.mean(self.fltr.apply(x1)[conn_indices], axis=1)
            rhoE2 = np.mean(self.fltr.apply(x2)[conn_indices], axis=1)

            u0_1 = u0 + du0dx[:, n] * dh
            u0_2 = u0 - du0dx[:, n] * dh

            l0_1 = l0 + dl0dx[n] * dh
            l0_2 = l0 - dl0dx[n] * dh

            u1_1 = u1 + dQ0dx[:, n] * dh / u1_norm
            u1_2 = u1 - dQ0dx[:, n] * dh / u1_norm

            # u1_1 = np.where(np.dot(u1, u1_1) < 0, -u1_1, u1_1)
            # u1_2 = np.where(np.dot(u1, u1_2) < 0, -u1_2, u1_2)

            u1_1_norm = u1_norm
            u1_2_norm = u1_norm

            # Q0_1 = Q0 + dQ0dx[:, n] * dh
            # Q0_2 = Q0 - dQ0dx[:, n] * dh

            # # Adjust the direction of Q1 and Q2 if necessary
            # Q0_1 = np.where(np.dot(Q0, Q0_1) < 0, -Q0_1, Q0_1)
            # Q0_2 = np.where(np.dot(Q0, Q0_2) < 0, -Q0_2, Q0_2)

            # # Normalize Q1 and Q2
            # u1_1_norm = np.linalg.norm(Q0_1)
            # u1_2_norm = np.linalg.norm(Q0_2)
            # u1_1 = Q0_1 / u1_1_norm
            # u1_2 = Q0_2 / u1_2_norm

            dl = l0_1 - l0_2
            du1 = u1_1 - u1_2

            ue0_1 = np.zeros((self.nelems, 8), dtype=rhoE1.dtype)
            ue0_2 = np.zeros((self.nelems, 8), dtype=rhoE2.dtype)
            ue0_1[:, ::2] = u0_1[2 * self.conn]
            ue0_1[:, 1::2] = u0_1[2 * self.conn + 1]
            ue0_2[:, ::2] = u0_2[2 * self.conn]
            ue0_2[:, 1::2] = u0_2[2 * self.conn + 1]

            ue1_1 = np.zeros((self.nelems, 8), dtype=rhoE1.dtype)
            ue1_2 = np.zeros((self.nelems, 8), dtype=rhoE2.dtype)
            ue1_1[:, ::2] = u1_1[2 * self.conn]
            ue1_1[:, 1::2] = u1_1[2 * self.conn + 1]
            ue1_2[:, ::2] = u1_2[2 * self.conn]
            ue1_2[:, 1::2] = u1_2[2 * self.conn + 1]

            # Compute the element stiffnesses
            if self.ptype_K == "simp":
                C1 = np.outer(rhoE1**self.p + self.rho0_K, self.C0)
                C2 = np.outer(rhoE2**self.p + self.rho0_K, self.C0)
            else:  # ramp
                C1 = np.outer(
                    rhoE1 / (1.0 + self.q * (1.0 - rhoE1)) + self.rho0_K, self.C0
                )
                C2 = np.outer(
                    rhoE2 / (1.0 + self.q * (1.0 - rhoE2)) + self.rho0_K, self.C0
                )

            C1 = C1.reshape((self.nelems, 3, 3))
            C2 = C2.reshape((self.nelems, 3, 3))

            Ke0_1 = np.zeros((self.nelems, 8, 8), dtype=rhoE1.dtype)
            Ke1_1 = np.zeros((self.nelems, 8, 8), dtype=rhoE1.dtype)
            Ke11_1 = np.zeros((self.nelems, 8, 8), dtype=rhoE1.dtype)

            Ge0_1 = np.zeros((self.nelems, 8, 8), dtype=rhoE1.dtype)
            Ge1_1 = np.zeros((self.nelems, 8, 8), dtype=rhoE1.dtype)

            Ke0_2 = np.zeros((self.nelems, 8, 8), dtype=rhoE2.dtype)
            Ke1_2 = np.zeros((self.nelems, 8, 8), dtype=rhoE2.dtype)
            Ke11_2 = np.zeros((self.nelems, 8, 8), dtype=rhoE2.dtype)

            Ge0_2 = np.zeros((self.nelems, 8, 8), dtype=rhoE2.dtype)
            Ge1_2 = np.zeros((self.nelems, 8, 8), dtype=rhoE2.dtype)

            dKe0 = np.zeros((self.nelems, 8, 8), dtype=rhoE1.dtype)
            dKe1 = np.zeros((self.nelems, 8, 8), dtype=rhoE1.dtype)
            dKe11 = np.zeros((self.nelems, 8, 8), dtype=rhoE1.dtype)

            dGe0 = np.zeros((self.nelems, 8, 8), dtype=rhoE1.dtype)
            dGe1 = np.zeros((self.nelems, 8, 8), dtype=rhoE1.dtype)

            for i in range(4):
                detJ = self.detJ[:, i]
                Be = self.Be[:, :, :, i]
                Te = self.Te[:, :, :, :, i]
                detJBei = detJBe[:, :, :, i]

                Be1_1 = np.zeros((self.nelems, 3, 8), dtype=rhoE1.dtype)
                Be1_2 = np.zeros((self.nelems, 3, 8), dtype=rhoE2.dtype)
                populate_nonlinear_strain_and_Be(Be, ue1_1, Be1_1)
                populate_nonlinear_strain_and_Be(Be, ue1_2, Be1_2)

                CKBe_1 = C1 @ Be
                CKBe_2 = C2 @ Be
                dCKBe = (C1 - C2) @ Be

                CKBe1_1 = C1 @ Be1_1
                CKBe1_2 = C2 @ Be1_2
                dCKBe1 = C1 @ Be1_1 - C2 @ Be1_2
                dBe1CBe1 = (
                    Be1_1.transpose(0, 2, 1) @ CKBe1_1
                    - Be1_2.transpose(0, 2, 1) @ CKBe1_2
                )

                Ke0_1 += detJBei @ CKBe_1
                Ke1_1 += detJBei @ CKBe1_1
                Ke11_1 += detJ.reshape(-1, 1, 1) * Be1_1.transpose(0, 2, 1) @ CKBe1_1

                Ke0_2 += detJBei @ CKBe_2
                Ke1_2 += detJBei @ CKBe1_2
                Ke11_2 += detJ.reshape(-1, 1, 1) * Be1_2.transpose(0, 2, 1) @ CKBe1_2

                dKe0 += detJBei @ dCKBe
                dKe1 += detJBei @ dCKBe1
                dKe11 += detJ.reshape(-1, 1, 1) * dBe1CBe1

                s0_1 = np.einsum("nik,nk -> ni", CKBe_1, ue0_1)
                G0e_1 = detJ.reshape(-1, 1, 1) * np.einsum("ni,nijl -> njl", s0_1, Te)
                Ge0_1[:, 0::2, 0::2] += G0e_1
                Ge0_1[:, 1::2, 1::2] += G0e_1

                s0_2 = np.einsum("nik,nk -> ni", CKBe_2, ue0_2)
                G0e_2 = np.einsum("n,ni,nijl -> njl", detJ, s0_2, Te)
                Ge0_2[:, 0::2, 0::2] += G0e_2
                Ge0_2[:, 1::2, 1::2] += G0e_2

                s1_1 = np.einsum("nik,nk -> ni", CKBe_1, ue1_1)
                G1e_1 = np.einsum("n,ni,nijl -> njl", detJ, s1_1, Te)
                Ge1_1[:, 0::2, 0::2] += G1e_1
                Ge1_1[:, 1::2, 1::2] += G1e_1

                s1_2 = np.einsum("nik,nk -> ni", CKBe_2, ue1_2)
                G1e_2 = np.einsum("n,ni,nijl -> njl", detJ, s1_2, Te)
                Ge1_2[:, 0::2, 0::2] += G1e_2
                Ge1_2[:, 1::2, 1::2] += G1e_2

                dG0e = detJ.reshape(-1, 1, 1) * np.einsum(
                    "ni,nijl -> njl", (s0_1 - s0_2), Te
                )
                dGe0[:, 0::2, 0::2] += dG0e
                dGe0[:, 1::2, 1::2] += dG0e

                dG1e = np.einsum("n,ni,nijl -> njl", detJ, (s1_1 - s1_2), Te)
                dGe1[:, 0::2, 0::2] += dG1e
                dGe1[:, 1::2, 1::2] += dG1e

            K0_1 = sparse.csc_matrix((Ke0_1.flatten(), (self.i, self.j)))
            K1_1 = sparse.csc_matrix((Ke1_1.flatten(), (self.i, self.j)))
            K11_1 = sparse.csc_matrix((Ke11_1.flatten(), (self.i, self.j)))
            G0_1 = sparse.csc_matrix((Ge0_1.flatten(), (self.i, self.j)))
            G1_1 = sparse.csc_matrix((Ge1_1.flatten(), (self.i, self.j)))

            K0_2 = sparse.csc_matrix((Ke0_2.flatten(), (self.i, self.j)))
            K1_2 = sparse.csc_matrix((Ke1_2.flatten(), (self.i, self.j)))
            K11_2 = sparse.csc_matrix((Ke11_2.flatten(), (self.i, self.j)))
            G0_2 = sparse.csc_matrix((Ge0_2.flatten(), (self.i, self.j)))
            G1_2 = sparse.csc_matrix((Ge1_2.flatten(), (self.i, self.j)))

            dK0 = sparse.csc_matrix((dKe0.flatten(), (self.i, self.j)))
            dK1 = sparse.csc_matrix((dKe1.flatten(), (self.i, self.j)))
            dK11 = sparse.csc_matrix((dKe11.flatten(), (self.i, self.j)))
            dG0 = sparse.csc_matrix((dGe0.flatten(), (self.i, self.j)))
            dG1 = sparse.csc_matrix((dGe1.flatten(), (self.i, self.j)))

            K0r_1 = self.reduce_matrix(K0_1)
            K1r_1 = self.reduce_matrix(K1_1)
            K11r_1 = self.reduce_matrix(K11_1)
            G0r_1 = self.reduce_matrix(G0_1)
            G1r_1 = self.reduce_matrix(G1_1)
            u1r_1 = self.reduce_vector(u1_1)

            K0r_2 = self.reduce_matrix(K0_2)
            K1r_2 = self.reduce_matrix(K1_2)
            K11r_2 = self.reduce_matrix(K11_2)
            G0r_2 = self.reduce_matrix(G0_2)
            G1r_2 = self.reduce_matrix(G1_2)
            u1r_2 = self.reduce_vector(u1_2)

            K0r = self.reduce_matrix(K0)
            K1r = self.reduce_matrix(K1)
            K11r = self.reduce_matrix(K11)
            G0r = self.reduce_matrix(G0)
            G1r = self.reduce_matrix(G1)
            u1r = self.reduce_vector(u1)

            dK0r = self.reduce_matrix(dK0)
            dK1r = self.reduce_matrix(dK1)
            dK11r = self.reduce_matrix(dK11)
            dG0r = self.reduce_matrix(dG0)
            dG1r = self.reduce_matrix(dG1)
            du1r = self.reduce_vector(du1)
            u2r = self.reduce_vector(u2)

            Ar_1 = K0r_1 + l0_1 * G0r_1
            Ar_2 = K0r_2 + l0_2 * G0r_2

            L_1 = (K0r_1 @ u1r_1).reshape(-1, 1)
            L_2 = (K0r_2 @ u1r_2).reshape(-1, 1)

            rhs_1 = np.hstack([-(G1r_1 + 0.5 * K1r_1) @ u1r_1, 0])
            rhs_2 = np.hstack([-(G1r_2 + 0.5 * K1r_2) @ u1r_2, 0])

            mat_1 = sparse.bmat([[Ar_1, L_1], [L_1.T, [0]]], format="csc")
            mat_2 = sparse.bmat([[Ar_2, L_2], [L_2.T, [0]]], format="csc")

            u2r_1 = sparse.linalg.spsolve(mat_1, rhs_1)[:-1]
            u2r_2 = sparse.linalg.spsolve(mat_2, rhs_2)[:-1]
            u2_1 = self.full_vector(u2r_1)
            u2_2 = self.full_vector(u2r_2)

            # # use cg to solve the linear system
            # u2r_1, _ = sparse.linalg.cg(mat_1, rhs_1, atol=1e-20,rtol=1e-20)
            # u2r_2, _ = sparse.linalg.cg(mat_2, rhs_2, atol=1e-20,rtol=1e-20)
            # u2r_1 = u2r_1[:-1]
            # u2r_2 = u2r_2[:-1]
            # u2_1 = self.full_vector(u2r_1)
            # u2_2 = self.full_vector(u2r_2)

            # dQr = -(dG1r + 0.5 * dK1r) @ u1r - (G1r + 0.5 * K1r) @ du1r
            # rhsr1 = -(dK0r + dl * G0r + l0 * dG0r) @ u2r + dQr
            # # rhs1r = self.reduce_vector(rhs1)

            # dLr = dK0r @ u1r + K0r @ du1r
            # rhsr2 = -dLr.T @ u2r

            # rhsr = np.hstack([rhsr1, rhsr2])

            # Ar = K0r + l0 * G0r
            # Lr = (K0r @ u1r).reshape(-1, 1)
            # mat = sparse.bmat([[Ar, Lr], [Lr.T, [0]]], format="csc")

            # rhs_mat[:, n] = rhsr
            # du2r = sparse.linalg.spsolve(mat, rhsr)[:-1]
            # du2 = self.full_vector(du2r)

            # # # use cg to solve the linear system
            # # du2rdx, _ = sparse.linalg.cg(mat, rhs, atol=1e-20,rtol=1e-20)
            # # du2 = self.full_vector(du2rdx[:-1])

            # # for ii in range(10):
            # #     print(u2_1[ii]- u2_2[ii], du2[ii])

            # ic(np.allclose(u2_1 - u2_2, du2, atol=1e-10))
            # exit()

            # D_1 = K1r_1 @ u1r_1
            # D_2 = K1r_2 @ u1r_2

            a_1 = 1.5 * u1_1_norm**2 * u1r_1.T @ K1r_1 @ u1r_1
            a_2 = 1.5 * u1_2_norm**2 * u1r_2.T @ K1r_2 @ u1r_2

            b_1 = u1_1_norm**2 * (
                u2r_1.T @ K1r_1 @ u1r_1
                + 2 * u1r_1.T @ K1r_1 @ u2r_1
                + 0.5 * u1r_1.T @ K11r_1 @ u1r_1
            )
            b_2 = u1_2_norm**2 * (
                u2r_2.T @ K1r_2 @ u1r_2
                + 2 * u1r_2.T @ K1r_2 @ u2r_2
                + 0.5 * u1r_2.T @ K11r_2 @ u1r_2
            )

            # du2 = self.full_vector(u2r_1 - u2r_2)
            # ic(dQ0.shape, Q0.shape)

            # Na = u1.T @ K1 @ u1
            # dNa = (u1.T @ dK1 + du1.T @ (K1 + K1.T)) @ u1
            # du1_norm = Q0.T @ dQ0
            # da = 3 * du1_norm * Na + 1.5 * u1_norm**2 * dNa
            # dadx[n] = da / dxi

            # Nb = u2r.T @ (K1r + 2 * K1r.T) @ u1r + 0.5 * u1r.T @ K11r @ u1r
            # dNb1 = (u2r.T @ (dK1r + 2 * dK1r.T) + 0.5 * u1r.T @ dK11r) @ u1r
            # dNb2 = du2r.T @ (K1r + 2 * K1r.T) @ u1r
            # dNb3 = (u2r.T @ (K1r + 2 * K1r.T) + u1r.T @ K11r) @ du1r
            # dN = dNb1 + dNb2 + dNb3
            # db = 2 * du1_norm * Nb + u1_norm**2 * dN
            # dbdx[n] = db / dxi

            # ic(np.allclose(a_2, a2))
            # ic(np.allclose(a_1, a1))

            # dK = sparse.csc_matrix((dKe.flatten(), (self.i, self.j)))
            # dK1dx[n] = (u.T @ dK @ v) / (2 * dh)

            # dK1dx[n] = (u.T @ (K1_1 - K1_2) @ v) / (2 * dh)
            # dbdx_temp[n] = (b_1 - b_2) / (2 * dh)
            # ic(a_1-a_2, da)
            # print(b_1 - b_2, db)
            # exit()
            dadx[n] = (a_1 - a_2) / (2 * dh)
            dbdx[n] = (b_1 - b_2) / (2 * dh)
            # dadx[n] = (a_1 - a_2) / (2 * dh)
            # dQdx[:, n] = (Q1 - Q2) / (2 * dh)

            _print_progress(n, self.nnodes, t0, "da/dx")

        return dadx, dbdx

    def check_derivative(self, rhoE, lam_c, u0, u1):
        # set random seed
        np.random.seed(1234)

        # set the perturbation size
        dh = 5e-3
        p = np.random.rand(self.x.size)
        p /= np.linalg.norm(p)
        x0 = self.x

        self.initialize()
        Q0 = self.Q[:, 0]
        Q0_norm = np.linalg.norm(Q0)
        self.initialize_koiter(Q0_norm)
        self.initialize_koiter_derivative()

        # Compute the perturbed design variables
        self.x = x0 + dh * p
        self.initialize()
        rhoE1, l1, Q1, u01 = self.rhoE, self.lam[0], self.Q[:, 0], self.u
        Q1 = np.where(np.dot(Q0, Q1) < 0, -Q1, Q1)
        a1, b1, u11, u21, K1_1, _, G1_1 = self.get_koiter_ab(
            rhoE1, l1, u01, Q1, Q0_norm
        )

        self.x = x0 - dh * p
        self.initialize()
        rhoE2, l2, Q2, u02 = self.rhoE, self.lam[0], self.Q[:, 0], self.u
        Q2 = np.where(np.dot(Q0, Q2) < 0, -Q2, Q2)
        a2, b2, u12, u22, K1_2, _, G1_2 = self.get_koiter_ab(
            rhoE2, l2, u02, Q2, Q0_norm
        )

        # Restore original x value to avoid side effects
        self.x = x0
        self.initialize()
        rhoE, l0, u0, Q0 = self.rhoE, self.lam[0], self.u, self.Q[:, 0]
        a, b, u1, u2, K1, K11, G1 = self.get_koiter_ab(rhoE, l0, u0, Q0)

        u1_norm = np.linalg.norm(Q0)
        u1 = Q0 / u1_norm

        #  Adjust eigenvectors' direction based on the dot product with Q0
        Q1 = np.where(np.dot(Q0, Q1) < 0, -Q1, Q1)
        Q2 = np.where(np.dot(Q0, Q2) < 0, -Q2, Q2)

        ic(l1, l2)

        dldx_cd = (l1 - l2) / (2 * dh)
        dldx = self.get_dldx()
        dldxp = np.dot(p, dldx)

        print("lam_c Exact: ", dldxp)
        print("lam_c CD   : ", dldx_cd)
        print("lam_c Error: ", (dldxp - dldx_cd) / dldx_cd, "\n")

        Q1_norm = np.linalg.norm(Q1)
        Q2_norm = np.linalg.norm(Q2)

        D1 = -l1 * u01.T @ K1_1 @ u11
        D2 = -l2 * u02.T @ K1_2 @ u12
        # D1 = u11.T @ K0_1 @ u11
        # D2 = u12.T @ K0_2 @ u12
        ic(D1, D2)
        D_cd = (D1 - D2) / (2 * dh)

        ic(self.Q[:, 0] @ K1 @ self.Q[:, 0])
        # self.initialize()
        self.initialize_adjoint()
        # Qb = -2 / u1_norm**4 * Q0  + 2 / u1_norm**2 * K0 @ Q0
        Qb = -2 / Q0_norm**4 * Q0
        self.Qrb[:, 0] = self.reduce_vector(Qb)
        self.finalize_adjoint()
        dnormdx = self.xb
        ans = np.dot(p, dnormdx)

        print("D ans  : ", ans)
        print("D CD   : ", D_cd)
        print("D Error: ", (ans - D_cd) / D_cd, "\n")

        # check if G1 is symmetric
        ic(np.allclose(G1.todense(), G1.todense().T))
        ic(np.allclose(K1.todense(), K1.todense().T))

        dK1dQ0 = self.get_dK1du(rhoE, Q0, Q0)

        # dK1dQ0_cd = self.get_dK1dQ0_cs(rhoE, l0, u0, Q0, Q0)
        G = self.get_stress_stiffness_matrix(self.rho, Q0)
        # ic(np.allclose(G.todense(), G.todense().T))

        # ic(np.allclose(dK1dQ0_cd, dK1dQ0))
        ic(np.allclose(G1 @ Q0 * u1_norm, dK1dQ0))

        # ic(np.allclose(G.todense(), dK1dQ0_cd.T))
        # ic(np.allclose(G.todense(), dK1dQ0_cd))

        # dK1dx_c = self.get_dK1dx(rhoE, Q0, Q0, Q0)
        # dK1dx_c = self.fltr.apply_gradient(dK1dx_c, x)

        self.initialize_adjoint()
        # Qb = (Q0.T @ (K1 + K1.T + G)) / u1_norm**2 - 2 * (
        #     Q0.T @ K1 @ Q0
        # ) * Q0.T / u1_norm**4
        Qb = Q0.T @ (K1 + K1.T)
        self.Qrb[:, 0] = self.reduce_vector(Qb)
        self.finalize_adjoint()
        dNdx1 = np.dot(p, self.xb)

        dK0dx, dG0dx, du0dx = self.get_dG0dx_dK0dx_du0dx_cs(
            x0, self.K0, self.G0, u0, Q0
        )
        dQ0dx, dl0dx = self.get_dQ0dx_dl0dx(x0, self.K0, self.G0, dK0dx, dG0dx, l0, Q0)

        dl0dx = self.get_dldx()
        # dQ0dx = self.get_exact_dQ0dx(Q0)

        # check if the computed dQ0dx is correct compared to dQ0dx0
        # ic(np.allclose(dQ0dx, dQ0dx0))

        # for i in range(5):
        #     print(dQ0dx0[0,i])
        #     print(dQ0dx[0,i])
        #     print((dQ0dx0[0,i] - dQ0dx[0,i]) / dQ0dx[0,i])
        #     print()
        # exit()

        # dK1dx, dadx, dQdx = self.get_derivative_cd(x, Q0, Q0, dh=dh)

        # ic(np.allclose(dQdx, dQdx_lee))

        # for i in range(5):
        #     print(f"{dQdx[10, i]:15.5e}{dQdx_lee[10, i]:15.5e}{dQdx[10, i]/dQdx_lee[10, i]:15.5e}")

        dK1dx_c = self.get_dK1dx_cd(x0, dQ0dx, Q0, Q0, Q0, dh=dh)
        dNdx1 = np.dot(p, Q0 @ (K1 + K1.T) @ dQ0dx + dK1dx_c)

        N1 = Q1.T @ K1_1 @ Q1
        N2 = Q2.T @ K1_2 @ Q2

        dNdx_cd = (N1 - N2) / (2 * dh)

        print("N ans  : ", dNdx1)
        print("N CD   : ", dNdx_cd)

        dadx, dbdx = self.get_derivative_cd(
            self.x,
            self.K0,
            K1,
            K11,
            self.G0,
            G1,
            du0dx,
            dQ0dx,
            dl0dx,
            l0,
            u0,
            Q0,
            u2,
            Q0,
            Q0,
            dh,
            Q0_norm,
        )
        dadx = np.dot(p, dadx)
        dbdx = np.dot(p, dbdx)

        print("a ans  : ", dadx)

        print("b ans  : ", dbdx)

        self.initialize_koiter(Q0_norm)
        self.initialize_koiter_derivative()

        dbdx_fast = self.get_dbdx(rhoE, l0, Q0, dl0dx, du0dx, dQ0dx)

        # print("a CD   : ", np.dot(p, self.dadx))
        print("a fast : ", np.dot(p, self.dadx))
        print("b fast : ", np.dot(p, dbdx_fast))

        dadx, dbdx = self.get_dadx_dbdx_cd(x0, du0dx, dl0dx, dQ0dx, u0, l0, Q0, dh=dh)

        dadx = np.dot(p, dadx)
        dbdx = np.dot(p, dbdx)
        print("a CD   : ", dadx)
        print("b CD   : ", dbdx)

        # print("a ans  : ", 1.5 * np.dot(p, Q0 @ (K1 + K1.T) @ dQ0dx + dK1dx_c))
        # print("a CD   : ", 1.5 * np.dot(self.xb + dK1dx_c, p))
        # print("a CD   : ", 1.5 * dNdx_cd)
        # print("a CD   : ", (a1 - a2) / (2 * dh))
        # print("a CD   : ", np.dot(p, dadx))

        dadx, dbdx = self.get_dadx_dbdx_cs(x0, du0dx, dl0dx, dQ0dx, u0, l0, Q0, dh=dh)

        dadx = np.dot(p, dadx)
        dbdx = np.dot(p, dbdx)
        print("a CS   : ", dadx)
        print("b CS   : ", dbdx)
        print("b CD   : ", (b1 - b2) / (2 * dh))

        ic(a, a1, a2)

        t0 = time.time()
        Gr = self.reduce_matrix(self.G0)
        Kr = self.reduce_matrix(self.K0)
        mu = -1 / l0

        Ar = Gr - mu * Kr
        Q0r = self.reduce_vector(Q0)
        Lr = (Kr @ Q0r).reshape(-1, 1)
        K1r = self.reduce_matrix(K1)
        mat = sparse.bmat([[Ar, -Lr], [-Lr.T, [0]]], format="csc")

        rhs = np.hstack([Q0r @ (K1r.T + 2 * K1r), 0])
        psi = sparse.linalg.spsolve(mat, -1.5 * rhs)
        psi0 = self.full_vector(psi[:-1])
        psi1 = psi[-1]

        dGdx1 = self.get_stress_stiffness_matrix_xuderiv(self.rhoE, self.u, psi0, Q0)
        dKdx1 = self.get_stiffness_matrix_deriv(self.rhoE, psi0, Q0)
        dKdx2 = self.get_stiffness_matrix_deriv(self.rhoE, Q0, Q0)
        dK1dx = self.get_dK1dx(self.rhoE, Q0, Q0, Q0)

        dadx = dGdx1 - mu * dKdx1 - 0.5 * psi1 * dKdx2 + 1.5 * dK1dx
        t1 = time.time()
        print("Time: ", t1 - t0)
        # dadx = 1.5 * Q0.T @ (K1 + K1.T) @ dQ0dx + 1.5 * dK1dx_c
        dadx = np.dot(p, dadx)
        print("a ans  : ", dadx)

        self.initialize_adjoint()
        dK1dQ0_cd = self.get_dK1dQ0_cd(rhoE, Q0, Q0, Q0, dh=1e-4)
        Qb = Q0.T @ (2 * K1 + K1.T)  # + dK1dQ0_cd
        self.Qrb[:, 0] = self.reduce_vector(Qb)
        self.finalize_adjoint()

        a_cd = np.dot(p, 1.5 * (self.xb + dK1dx))

        self.initialize_adjoint()
        dK1dQ0_cs = self.get_dK1dQ0_cs(rhoE, K1, Q0, Q0, Q0, dh=1e-30)
        Qb = Q0.T @ (K1 + K1.T) + dK1dQ0_cs
        self.Qrb[:, 0] = self.reduce_vector(Qb)
        self.finalize_adjoint()

        a_cs = np.dot(p, 1.5 * (self.xb + dK1dx))

        print("a cs   : ", a_cs)
        print("a cd   : ", a_cd)

        exit()

        G_1 = self.get_stress_stiffness_matrix(self.rho, u1)
        G_2 = self.get_stress_stiffness_matrix(self.rho, u2)

        # Check if G_1 and G_2 are symmetric
        ic(np.allclose(G_1.todense(), G_1.todense().T))
        ic(np.allclose(G_2.todense(), G_2.todense().T))

        aa = G_1 @ u2
        bb = G_2 @ u1
        # check if G_1 @ u2 = G_2 @ u1
        ic(np.allclose(aa, bb))

        u1Gu2 = self.get_dK1dQ0_cs(self.rhoE, K1, Q0, u1, u2) * u1_norm

        ic(aa.shape, bb.shape, u1Gu2.shape)

        ic(np.allclose(u1Gu2, aa))
        ic(np.allclose(u1Gu2, bb))

        for i in range(10):
            print(f"{u1Gu2[i]:15.5e}{aa[i]:15.5e}{bb[i]:15.5e}")

        exit()

        t0 = time.time()
        papx = self.get_papx_cs(self.x, a, Q0)

        # use linspace gererate dh from 1e+1 to 1e-10
        dh0 = 6
        dh1 = -20
        dhn = 4 * (dh0 - dh1 + 1)
        dh = np.logspace(dh0, dh1, num=dhn)

        a_cs = np.zeros(dhn)
        a_cd = np.zeros(dhn)

        # for di in range(dhn):
        #     ic(dh[di])
        #     self.initialize_adjoint()
        #     dK1dQ0_cs = self.get_dK1dQ0_cs(rhoE, K1, Q0, Q0, Q0, dh=dh[di])
        #     Qb = Q0.T @ (K1 + K1.T) + dK1dQ0_cs
        #     self.Qrb[:, 0] = self.reduce_vector(Qb)
        #     self.finalize_adjoint()

        #     a_cs[di] = np.dot(p, 1.5 * self.xb + papx)
        #     ic(a_cs[di])

        #     self.initialize_adjoint()
        #     dK1dQ0_cd = self.get_dK1dQ0_cd(rhoE, Q0, Q0, Q0, dh=dh[di])
        #     Qb = Q0.T @ (K1 + K1.T) + dK1dQ0_cd
        #     self.Qrb[:, 0] = self.reduce_vector(Qb)
        #     self.finalize_adjoint()

        #     a_cd[di] = np.dot(p, 1.5 * self.xb + papx)
        #     ic(a_cd[di])

        # # save the data
        # np.save("./output/a_cs.npy", a_cs)
        # np.save("./output/a_cd.npy", a_cd)

        # read the data
        a_cs = np.load("./output/a_cs.npy")
        a_cd = np.load("./output/a_cd.npy")

        self.initialize_adjoint()
        dK1dQ0_cd = self.get_dK1dQ0_cd(rhoE, Q0, Q0, Q0, dh=1e-4)
        Qb = Q0.T @ (K1 + K1.T) + dK1dQ0_cd
        self.Qrb[:, 0] = self.reduce_vector(Qb)
        self.finalize_adjoint()

        a_ans1 = np.dot(p, 1.5 * self.xb + papx)

        self.initialize_adjoint()
        dK1dQ0_cs = self.get_dK1dQ0_cs(rhoE, K1, Q0, Q0, Q0, dh=1e-100)
        Qb = Q0.T @ (K1 + K1.T) + dK1dQ0_cs
        self.Qrb[:, 0] = self.reduce_vector(Qb)
        self.finalize_adjoint()

        a_ans2 = np.dot(p, 1.5 * self.xb + papx)

        dK1dx = self.get_dK1dx(rhoE, Q0, Q0, Q0)
        dK1du = self.get_dK1du(rhoE, Q0, Q0)
        dK1dQ0 = dK1du / u1_norm

        self.initialize_adjoint()
        Qb = Q0.T @ (K1 + K1.T) + G1 @ Q0
        self.Qrb[:, 0] = self.reduce_vector(Qb)
        self.finalize_adjoint()

        a_ans3 = np.dot(p, (1.5 * self.xb + 1.5 * dK1dx))

        a_ans = [a_ans1, a_ans2, a_ans3]
        title = [
            f"baseline: cd = 1e-4",
            f"baseline: cs = 1e-100",
            f"baseline: analytical",
        ]

        fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), tight_layout=True)
        for i in range(3):
            error_cd = np.abs((a_cd - a_ans[i]) / a_ans[i])
            error_cs = np.abs((a_cs - a_ans[i]) / a_ans[i])
            ax[i].plot(dh, error_cs, label="CS")
            ax[i].plot(dh, error_cd, label="CD")
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            ax[i].set_xlabel("dh")
            ax[i].set_ylabel("Relative Error")
            ax[i].invert_xaxis()
            ax[i].set_title(title[i])
            ax[i].legend()
        plt.savefig("./output/error.pdf")
        exit()

        t3 = time.time()

        print("a CS   : ", a_cs)
        print("a CD   : ", a_cd)
        print("a ans  : ", a_ans)
        print("a error: ", (a_cs - a_ans) / a_ans)
        print("t CS   : ", t1 - t0)
        print("t ans  : ", t3 - t2)

        # for i in range(2 * self.nnodes):
        #     print(
        #         f"{dK1dQ0_cd[i]:15.5e}{dK1du[i]/u1_norm:15.5e}{(dK1dQ0_cd[i] - dK1du[i]/u1_norm)/dK1dQ0_cd[i]:15.5e}"
        #     )

        # for i in range(self.nnodes):
        #     print(
        #         f"{papx[i]:15.5e}{1.5 * dK1dx[i]:15.5e}{(papx[i] - 1.5 * dK1dx[i])/papx[i]:15.5e}"
        #     )
        exit()

        dal0dx = self.get_dal0dx(l0, a, dl0dx, dadx)
        ans = (a1 * l1 - a2 * l2) / (2 * dh)
        print("al ans  :", ans)
        print("al CD   :", np.dot(p, dal0dx))
        print("al Error:", (ans - np.dot(p, dal0dx)) / ans)

        xi = 1e-3
        a = -0.010895207750829469
        lam_s1 = self.get_lam_s(l1, a1, xi)
        lam_s2 = self.get_lam_s(l2, a2, xi)
        ans = (lam_s1 - lam_s2) / (2 * dh)
        ic(lam_s1, lam_s2)

        dlam_sdx = self.get_dlamsdx(l0, a, dl0dx, dadx, xi)
        dlam_sdx = np.dot(p, dlam_sdx)

        print("lam_s ans  :", ans)
        print("lam_s CD   :", dlam_sdx)
        print("lam_s Error:", (ans - dlam_sdx) / ans)

        flam_s1 = self.get_flam_s(l1, a1, xi)
        flam_s2 = self.get_flam_s(l2, a2, xi)
        ans = (flam_s1 - flam_s2) / (2 * dh)
        ic(flam_s1, flam_s2)

        dflam_sdx = self.get_dflamsdx(a, dadx, xi)
        dflam_sdx = np.dot(p, dflam_sdx)

        print("flam_s ans  :", ans)
        print("flam_s CD   :", dflam_sdx)
        print("flam_s Error:", (ans - dflam_sdx) / ans)

        return

    def check_koiter_ab(self, lam_c, a, b, u0, u1, u2):
        K0 = self.K0
        K1 = self.K1
        G0 = self.G0
        G1 = self.G1
        K11 = self.K11

        K0r = self.reduce_matrix(self.K0)
        K1r = self.reduce_matrix(self.K1)
        G0r = self.reduce_matrix(self.G0)
        G1r = self.reduce_matrix(self.G1)
        K11r = self.reduce_matrix(self.K11)

        u0r = self.reduce_vector(u0)
        u1r = self.reduce_vector(u1)
        u2r = self.reduce_vector(u2)
        fr = self.reduce_vector(self.f)

        # check if u1 is normalised
        ic(np.allclose(np.linalg.norm(u1), 1))

        # check Ku0 = f
        ic(np.allclose(self.Kr @ u0r, fr))

        # check if K0 = K, G0 = G
        K = self.get_stiffness_matrix(self.rho)
        G = self.get_stress_stiffness_matrix(self.rho, u0)
        ic(np.allclose((K0 - K).data, 0))
        ic(np.allclose((G0 - G).data, 0))

        # check if linear eigenvalue is correct
        u1r = self.reduce_vector(u1)
        ic(np.allclose((K0r + lam_c * G0r) @ u1r, 0))

        # check the orthogonality condition
        # check if stress_1 * l1(u2) = 0
        ic(u1r.T @ K0r @ u2r)
        ic(u1.T @ K0 @ u2)
        # check if stress_0 * l11(u1, u2) = 0
        ic(u0r.T @ K1r @ u2r)
        ic(u0.T @ K1 @ u2)

        # check if a can be change but get the same result
        a = sparse.csc_matrix(([-1], ([0], [0])), shape=(1, 1))
        Ar = K0r + lam_c * G0r
        L = (K0r @ u1r).reshape(-1, 1)
        A = sparse.bmat([[Ar, L], [L.T, a]], format="csc")
        R = -(G1r + 0.5 * K1r) @ u1r
        x = sparse.linalg.spsolve(A, np.hstack([R, 0]))
        ic(np.allclose(x[:-1], u2r, atol=1e-3))

        # check if - -lam_c * u0 @ K1 @ u1 = u1 @ K0 @ u1
        a = -lam_c * u0.T @ K1 @ u1
        b = u1.T @ K0 @ u1
        ic(a, b)

        # check if u1 @ K1 @ u2 != u2 @ K1 @ u1, since K1 is not symmetric
        ic(np.allclose(u1.T @ K1 @ u2, u2.T @ K1 @ u1))

        # check a and b
        d = u1r.T @ K0r @ u1r
        a = 1.5 * (u1r.T @ K1r @ u1r) / d
        b = (u2r.T @ K1r @ u1r + 2 * u1r.T @ K1r @ u2r + 0.5 * u1r.T @ K11r @ u1r) / d
        ic(a, b)

        return

    def intital_Be_and_Te(self):
        # Compute gauss points
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        # Compute Be and Te, detJ
        self.Be = np.zeros((self.nelems, 3, 8, 4))
        self.Te = np.zeros((self.nelems, 3, 4, 4, 4))
        self.detJ = np.zeros((self.nelems, 4))

        self.detJ_kokkos = np.zeros((2, 2, self.nelems))
        self.Be_kokkos = np.zeros((2, 2, self.nelems, 3, 8))
        self.Te_kokkos = np.zeros((2, 2, self.nelems, 3, 4, 4))

        for j in range(2):
            for i in range(2):
                xi, eta = gauss_pts[i], gauss_pts[j]
                index = 2 * j + i
                Bei = self.Be[:, :, :, index]
                Tei = self.Te[:, :, :, :, index]

                self.detJ[:, index] = populate_Be_and_Te(
                    self.nelems, xi, eta, xe, ye, Bei, Tei
                )

                self.detJ_kokkos[i, j, :] = self.detJ[:, index]
                self.Be_kokkos[i, j, :, :, :] = Bei
                self.Te_kokkos[i, j, :, :, :, :] = Tei

        return

    def intital_stress_stiffness_matrix_deriv(self, rhoE, Te, detJ, psi, phi):

        # Compute the element stiffnesses
        if self.ptype_G == "simp":
            C = np.outer(rhoE**self.p + self.rho0_G, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_G, self.C0)

        self.C = C.reshape((self.nelems, 3, 3))

        # Compute the element-wise values of psi and phi
        psie = np.zeros((self.nelems, 8) + psi.shape[1:])
        psie[:, ::2, ...] = psi[2 * self.conn, ...]
        psie[:, 1::2, ...] = psi[2 * self.conn + 1, ...]

        phie = np.zeros((self.nelems, 8) + phi.shape[1:])
        phie[:, ::2, ...] = phi[2 * self.conn, ...]
        phie[:, 1::2, ...] = phi[2 * self.conn + 1, ...]

        pp0 = psie[:, ::2] @ phie[:, ::2].transpose(0, 2, 1)
        pp1 = psie[:, 1::2] @ phie[:, 1::2].transpose(0, 2, 1)

        se = np.einsum("nijlm,njl -> nim", Te, (pp0 + pp1))
        dfds = detJ[:, np.newaxis, :] * se

        return dfds

    def get_stress_stiffness_matrix_uderiv_tensor(self, dfds, Be):

        Cdfds = self.C @ dfds
        dfdue = np.einsum("nijm,nim -> nj", Be, Cdfds)

        dfdu = np.zeros(2 * self.nnodes)
        np.add.at(dfdu, 2 * self.conn, dfdue[:, 0::2])
        np.add.at(dfdu, 2 * self.conn + 1, dfdue[:, 1::2])

        return dfdu

    def get_stress_stiffness_matrix_xderiv_tensor(self, rhoE, u, dfds, Be):

        # The element-wise variables
        ue = np.zeros((self.nelems, 8))
        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        dfds = np.einsum("nim, ij -> njm", dfds, self.C0)
        dfdrhoE = np.einsum("njm,njkm,nk -> n", dfds, Be, ue)

        if self.ptype_G == "simp":
            dfdrhoE[:] *= self.p * rhoE ** (self.p - 1)
        else:  # ramp
            dfdrhoE[:] *= (2.0 + self.q) / (1.0 + (self.q + 1.0) * (1.0 - rhoE)) ** 2

        dfdrho = np.zeros(self.nnodes)
        np.add.at(dfdrho, self.conn, dfdrhoE[:, np.newaxis])
        dfdrho *= 0.25

        return self.fltr.apply_gradient(dfdrho, self.x)

    def get_stress_stiffness_matrix_uderiv(self, rhoE, psi, phi):
        """
        Compute the derivative of psi^{T} * G(u, x) * phi using the adjoint method.

        Note "solver" returns the solution of the system of equations

        K * sol = rhs

        Given the right-hand-side rhs. ie. sol = solver(rhs)
        """

        # Compute the element stiffnesses
        if self.ptype_G == "simp":
            C = np.outer(rhoE**self.p + self.rho0_G, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_G, self.C0)

        C = C.reshape((self.nelems, 3, 3))

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Compute the element-wise values of psi and phi
        psie = np.zeros((self.nelems, 8) + psi.shape[1:])
        psie[:, ::2, ...] = psi[2 * self.conn, ...]
        psie[:, 1::2, ...] = psi[2 * self.conn + 1, ...]

        phie = np.zeros((self.nelems, 8) + phi.shape[1:])
        phie[:, ::2, ...] = phi[2 * self.conn, ...]
        phie[:, 1::2, ...] = phi[2 * self.conn + 1, ...]

        dfdue = np.zeros((self.nelems, 8))

        # Compute the element stress stiffness matrix
        Be = np.zeros((self.nelems, 3, 8))
        Te = np.zeros((self.nelems, 3, 4, 4))

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        if psi.ndim == 2 and phi.ndim == 2 and self.pp is None:
            pp0 = psie[:, ::2] @ phie[:, ::2].transpose(0, 2, 1)
            pp1 = psie[:, 1::2] @ phie[:, 1::2].transpose(0, 2, 1)
            self.pp = pp0 + pp1

        for xi, eta in [(xi, eta) for xi in gauss_pts for eta in gauss_pts]:
            detJ = populate_Be_and_Te(self.nelems, xi, eta, xe, ye, Be, Te)

            if psi.ndim == 1 and phi.ndim == 1:
                se0 = np.einsum("nijl,nj,nl -> ni", Te, psie[:, ::2], phie[:, ::2])
                se1 = np.einsum("nijl,nj,nl -> ni", Te, psie[:, 1::2], phie[:, 1::2])
                se = se0 + se1

            elif psi.ndim == 2 and phi.ndim == 2:
                se = np.einsum("nijl,njl -> ni", Te, self.pp)

            # Add contributions to the derivative w.r.t. u
            dfds = detJ[:, np.newaxis] * se
            BeC = np.matmul(Be.transpose(0, 2, 1), C)
            dfdue += np.einsum("njk,nk -> nj", BeC, dfds)

        dfdu = np.zeros(2 * self.nnodes)
        np.add.at(dfdu, 2 * self.conn, dfdue[:, 0::2])
        np.add.at(dfdu, 2 * self.conn + 1, dfdue[:, 1::2])

        return dfdu

    def get_stress_stiffness_matrix_xderiv(self, rhoE, u, psi, phi):
        """
        Compute the derivative of psi^{T} * G(u, x) * phi using the adjoint method.

        Note "solver" returns the solution of the system of equations

        K * sol = rhs

        Given the right-hand-side rhs. ie. sol = solver(rhs)
        """

        dfdrhoE = np.zeros(self.nelems)

        # Compute the element stiffnesses
        if self.ptype_G == "simp":
            C = np.outer(rhoE**self.p + self.rho0_G, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_G, self.C0)

        C = C.reshape((self.nelems, 3, 3))

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # The element-wise variables
        ue = np.zeros((self.nelems, 8))
        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        # Compute the element-wise values of psi and phi
        psie = np.zeros((self.nelems, 8) + psi.shape[1:])
        psie[:, ::2, ...] = psi[2 * self.conn, ...]
        psie[:, 1::2, ...] = psi[2 * self.conn + 1, ...]

        phie = np.zeros((self.nelems, 8) + phi.shape[1:])
        phie[:, ::2, ...] = phi[2 * self.conn, ...]
        phie[:, 1::2, ...] = phi[2 * self.conn + 1, ...]

        # Compute the element stress stiffness matrix
        Be = np.zeros((self.nelems, 3, 8))
        Te = np.zeros((self.nelems, 3, 4, 4))

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        if psi.ndim == 2 and phi.ndim == 2 and self.pp is None:
            pp0 = psie[:, ::2] @ phie[:, ::2].transpose(0, 2, 1)
            pp1 = psie[:, 1::2] @ phie[:, 1::2].transpose(0, 2, 1)
            self.pp = pp0 + pp1

        for xi, eta in [(xi, eta) for xi in gauss_pts for eta in gauss_pts]:
            detJ = populate_Be_and_Te(self.nelems, xi, eta, xe, ye, Be, Te)

            if psi.ndim == 1 and phi.ndim == 1:
                se0 = np.einsum("nijl,nj,nl -> ni", Te, psie[:, ::2], phie[:, ::2])
                se1 = np.einsum("nijl,nj,nl -> ni", Te, psie[:, 1::2], phie[:, 1::2])
                se = se0 + se1

            elif psi.ndim == 2 and phi.ndim == 2:
                se = np.einsum("nijl,njl -> ni", Te, self.pp)

            dfds = detJ[:, np.newaxis] * se @ self.C0
            dfdrhoE += np.einsum("nj,njk,nk -> n", dfds, Be, ue)

        if self.ptype_G == "simp":
            dfdrhoE[:] *= self.p * rhoE ** (self.p - 1)
        else:  # ramp
            dfdrhoE[:] *= (2.0 + self.q) / (1.0 + (self.q + 1.0) * (1.0 - rhoE)) ** 2

        dfdrho = np.zeros(self.nnodes)
        np.add.at(dfdrho, self.conn, dfdrhoE[:, np.newaxis])
        dfdrho *= 0.25

        return self.fltr.apply_gradient(dfdrho, self.x)

    def get_stress_stiffness_matrix_deriv(self, rhoE, u, psi, phi, solver):
        """
        Compute the derivative of psi^{T} * G(u, x) * phi using the adjoint method.

        Note "solver" returns the solution of the system of equations

        K * sol = rhs

        Given the right-hand-side rhs. ie. sol = solver(rhs)
        """

        dfdC = np.zeros((self.nelems, 3, 3))

        # Compute the element stiffnesses
        if self.ptype_G == "simp":
            C = np.outer(rhoE**self.p + self.rho0_G, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_G, self.C0)

        C = C.reshape((self.nelems, 3, 3))

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # The element-wise variables
        ue = np.zeros((self.nelems, 8))
        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        # Compute the element-wise values of psi and phi
        psie = np.zeros((self.nelems, 8))
        psie[:, ::2] = psi[2 * self.conn]
        psie[:, 1::2] = psi[2 * self.conn + 1]

        phie = np.zeros((self.nelems, 8))
        phie[:, ::2] = phi[2 * self.conn]
        phie[:, 1::2] = phi[2 * self.conn + 1]

        dfdue = np.zeros((self.nelems, 8))

        # Compute the element stress stiffness matrix
        Be = np.zeros((self.nelems, 3, 8))
        Te = np.zeros((self.nelems, 3, 4, 4))

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                detJ = populate_Be_and_Te(self.nelems, xi, eta, xe, ye, Be, Te)

                # Compute the derivative of the stress w.r.t. u
                dfds = np.einsum(
                    "n,nijl,nj,nl -> ni", detJ, Te, psie[:, ::2], phie[:, ::2]
                ) + np.einsum(
                    "n,nijl,nj,nl -> ni", detJ, Te, psie[:, 1::2], phie[:, 1::2]
                )

                # Add up contributions to d( psi^{T} * G(x, u) * phi ) / du
                dfdue += np.einsum("nij,nik,nk -> nj", Be, C, dfds)

                # Add contributions to the derivative w.r.t. C
                dfdC += np.einsum("ni,njk,nk -> nij", dfds, Be, ue)

        dfdu = np.zeros(2 * self.nnodes)
        np.add.at(dfdu, 2 * self.conn, dfdue[:, 0::2])
        np.add.at(dfdu, 2 * self.conn + 1, dfdue[:, 1::2])

        dfdur = self.reduce_vector(dfdu)

        # Compute the adjoint for K * adj = d ( psi^{T} * G(u, x) * phi ) / du
        adjr = solver(dfdur)
        adj = self.full_vector(adjr)

        dfdrhoE = np.zeros(self.nelems)
        for i in range(3):
            for j in range(3):
                dfdrhoE[:] += self.C0[i, j] * dfdC[:, i, j]

        if self.ptype_G == "simp":
            dfdrhoE[:] *= self.p * rhoE ** (self.p - 1)
        else:  # ramp
            dfdrhoE[:] *= (2.0 + self.q) / (1.0 + (self.q + 1.0) * (1.0 - rhoE)) ** 2

        dfdrho = np.zeros(self.nnodes)

        for i in range(4):
            np.add.at(dfdrho, self.conn[:, i], dfdrhoE)
        dfdrho *= 0.25

        # Compute the derivative of the stiffness matrix w.r.t. rho
        dfdrho -= self.get_stiffness_matrix_deriv(rhoE, adj, u)

        return dfdrho

    def eval_area(self):
        return np.sum(self.detJ.reshape(-1) * np.tile(self.rhoE, 4))

    def eval_area_gradient(self):
        dfdrhoE = np.sum(self.detJ, axis=1)

        dfdrho = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(dfdrho, self.conn[:, i], dfdrhoE)
        dfdrho *= 0.25

        return self.fltr.apply_gradient(dfdrho, self.x)

    def reduce_vector(self, forces):
        """
        Eliminate essential boundary conditions from the vector
        """
        return forces[self.reduced]

    def reduce_matrix(self, matrix):
        """
        Eliminate essential boundary conditions from the matrix
        """
        temp = matrix[self.reduced, :]
        return temp[:, self.reduced]

    def full_vector(self, vec):
        """
        Transform from a reduced vector without dirichlet BCs to the full vector
        """
        temp = np.zeros((self.nvars,) + vec.shape[1:], dtype=vec.dtype)
        temp[self.reduced, ...] = vec[:, ...]
        return temp

    def full_matrix(self, mat):
        """
        Transform from a reduced matrix without dirichlet BCs to the full matrix
        """
        temp = np.zeros((self.nvars, self.nvars), dtype=mat.dtype)
        for i in range(len(self.reduced)):
            for j in range(len(self.reduced)):
                temp[self.reduced[i], self.reduced[j]] = mat[i, j]
        return temp

    def _init_profile(self):
        self.profile = {}
        self.profile["nnodes"] = self.nnodes
        self.profile["nelems"] = self.nelems
        self.profile["solver_type"] = self.solver_type
        self.profile["adjoint_method"] = self.adjoint_method
        self.profile["adjoint_options"] = self.adjoint_options
        self.profile["N"] = self.N
        self.profile["E"] = self.E
        self.profile["nu"] = self.nu
        self.profile["density"] = self.density
        self.profile["p"] = self.p
        self.profile["eig_atol"] = self.eig_atol
        self.profile["ftype"] = self.fltr.ftype
        self.profile["r0"] = self.fltr.r0

        return

    def solve_eigenvalue_problem(self, store=False):
        """
        Compute the smallest buckling load factor BLF
        """
        t0 = time.time()

        self.K0 = self.get_stiffness_matrix(self.rho)
        self.Kr = self.reduce_matrix(self.K0)

        # Compute the solution path
        self.fr = self.reduce_vector(self.f)
        self.Kfact = linalg.factorized(self.Kr.tocsc())
        ur = self.Kfact(self.fr)
        self.u = self.full_vector(ur)

        # Find the gemoetric stiffness matrix
        self.G0 = self.get_stress_stiffness_matrix(self.rho, self.u)
        self.Gr = self.reduce_matrix(self.G0)

        t1 = time.time()
        self.profile["matrix assembly time"] += t1 - t0

        # Find the eigenvalues closest to zero. This uses a shift and
        # invert strategy around sigma = 0, which means that the largest
        # magnitude values are closest to zero.
        for i in range(self.cost):
            if self.N >= self.nvars:
                mu, self.Q = eigh(self.Gr.todense(), self.Kr.todense())
                mu, self.Qr = mu[: self.N], self.Qr[:, : self.N]
            else:
                self.profile["sigma"] = self.sigma if i == 0 else None

                # Compute the shifted operator
                mat = self.Kr + self.sigma * self.Gr
                mat = mat.tocsc()
                self.factor = SpLuOperator(mat)
                self.factor.count = 0

                if self.solver_type == "IRAM":
                    if self.m is None:
                        self.m = max(2 * self.N + 1, 60)
                    self.eig_solver = IRAM(
                        N=self.N, m=self.m, eig_atol=self.eig_atol, mode="buckling"
                    )
                    mu, self.Qr = self.eig_solver.solve(
                        self.Gr, self.Kr, self.factor, self.sigma
                    )
                else:
                    if self.m is None:
                        self.m = max(3 * self.N + 1, 60)
                    self.eig_solver = BasicLanczos(
                        N=self.N,
                        m=self.m,
                        eig_atol=self.eig_atol,
                        tol=self.tol,
                        mode="buckling",
                    )
                    mu, self.Qr = self.eig_solver.solve(
                        self.Gr,
                        self.Kr,
                        self.factor,
                        self.sigma,
                    )

                    if store:
                        self.profile["eig_res"] = self.eig_solver.eig_res.tolist()

                self.profile["solve preconditioner count"] += (
                    self.factor.count if i == 0 else 0
                )

        t2 = time.time()
        t = (t2 - t1) / self.cost

        self.profile["eigenvalue solve time"] += t
        logging.info("Eigenvalue solve time: %5.2f s" % t)

        self.profile["m"] = self.m
        self.profile["eig_solver.m"] = str(self.eig_solver.m)
        logging.info("eig_solver.m = %d" % self.eig_solver.m)

        self.BLF = mu[: self.N]

        # project the eigenvectors back to the full space
        Q = np.zeros((self.nvars, self.N), dtype=self.rhoE.dtype)
        Q[self.reduced, ...] = self.Qr[:, ...]

        return mu, Q

    def compliance(self):
        return self.f.dot(self.u)

    def compliance_derivative(self):
        dfdrho = -1.0 * self.get_stiffness_matrix_deriv(self.rhoE, self.u, self.u)
        return dfdrho

    def eval_ks_buckling(self, ks_rho=160.0):
        mu = 1 / self.BLF
        c = max(mu)
        eta = np.exp(ks_rho * (mu - c))
        ks_min = c + np.log(np.sum(eta)) / ks_rho
        return ks_min

    def eval_ks_buckling_derivative(self, ks_rho=160.0):
        t0 = time.time()
        mu = 1 / self.BLF
        c = max(mu)
        eta = np.exp(ks_rho * (mu - c))
        eta = eta / np.sum(eta)

        dfdx = np.zeros(self.x.shape)
        if self.deriv_type == "vector":
            for i in range(self.N):
                dKdx = self.get_stiffness_matrix_deriv(
                    self.rhoE, self.Q[:, i], self.Q[:, i]
                )
                dGdx = self.get_stress_stiffness_matrix_xderiv(
                    self.rhoE, self.u, self.Q[:, i], self.Q[:, i]
                )

                dGdu = self.get_stress_stiffness_matrix_uderiv(
                    self.rhoE, self.Q[:, i], self.Q[:, i]
                )
                dGdur = self.reduce_vector(dGdu)
                adjr = -self.Kfact(dGdur)
                adj = self.full_vector(adjr)

                dGdx += self.get_stiffness_matrix_deriv(self.rhoE, adj, self.u)
                dfdx -= eta[i] * (dGdx + mu[i] * dKdx)

        elif self.deriv_type == "tensor":
            eta_Q = (eta[:, np.newaxis] * self.Q.T).T
            eta_mu_Q = (eta[:, np.newaxis] * mu[:, np.newaxis] * self.Q.T).T

            dKdx = self.get_stiffness_matrix_deriv(self.rhoE, eta_mu_Q, self.Q)

            dfds = self.intital_stress_stiffness_matrix_deriv(
                self.rhoE, self.Te, self.detJ, eta_Q, self.Q
            )
            dGdu = self.get_stress_stiffness_matrix_uderiv_tensor(dfds, self.Be)
            dGdur = self.reduce_vector(dGdu)
            adjr = -self.Kfact(dGdur)
            adj = self.full_vector(adjr)

            dGdx = self.get_stress_stiffness_matrix_xderiv_tensor(
                self.rhoE, self.u, dfds, self.Be
            )
            dGdx += self.get_stiffness_matrix_deriv(self.rhoE, adj, self.u)
            dfdx -= dGdx + dKdx

        t1 = time.time()
        self.profile["total derivative time"] += t1 - t0

        return dfdx

    def get_eigenvector_aggregate(self, rho, node, mode="tanh"):
        if mode == "exp":
            eta = np.exp(-rho * (self.lam - np.min(self.lam)))
        else:
            lam_a = 0.0
            lam_b = 50.0

            a = np.tanh(rho * (self.lam - lam_a))
            b = np.tanh(rho * (self.lam - lam_b))
            eta = a - b

        # Normalize the weights
        eta = eta / np.sum(eta)

        # print(eta)

        h = 0.0
        for i in range(self.N):
            h += eta[i] * np.dot(self.Q[node, i], self.Q[node, i])

        return h

    def add_eigenvector_aggregate_derivative(self, hb, rho, node, mode="tanh"):
        if mode == "exp":
            eta = np.exp(-rho * (self.lam - np.min(self.lam)))
        else:
            lam_a = 0.0
            lam_b = 50.0

            a = np.tanh(rho * (self.lam - lam_a))
            b = np.tanh(rho * (self.lam - lam_b))
            eta = a - b

        # Normalize the weights
        eta = eta / np.sum(eta)

        h = 0.0
        for i in range(self.N):
            h += eta[i] * np.dot(self.Q[node, i], self.Q[node, i])

        Qb = np.zeros(self.Q.shape)
        for i in range(self.N):
            Qb[node, i] += 2.0 * hb * eta[i] * self.Q[node, i]
            self.Qrb[:, i] += Qb[self.reduced, i]

            if mode == "exp":
                self.lamb[i] -= (
                    hb * rho * eta[i] * (np.dot(self.Q[node, i], self.Q[node, i]) - h)
                )
            else:
                self.lamb[i] -= (
                    hb
                    * rho
                    * eta[i]
                    * (a[i] + b[i])
                    * (np.dot(self.Q[node, i], self.Q[node, i]) - h)
                )

        return

    def KSmax(self, q, ks_rho):
        c = np.max(q)
        eta = np.exp(ks_rho * (q - c))
        ks_max = c + np.log(np.sum(eta)) / ks_rho
        return ks_max

    def eigenvector_aggregate_magnitude(self, rho, node):
        # Tanh aggregate
        lam_a = 0.0
        lam_b = 1000.0
        a = np.tanh(rho * (self.lam - lam_a))
        b = np.tanh(rho * (self.lam - lam_b))
        eta = a - b

        # Normalize the weights
        eta = eta / np.sum(eta)

        h = 0.0
        for i in range(self.N):
            h += eta[i] * self.Q[node, i] ** 2

        return h, eta, a, b

    def get_eigenvector_aggregate_max(self, rho, node):
        h, _, _, _ = self.eigenvector_aggregate_magnitude(rho, node)
        h = self.KSmax(h, rho)
        return h

    def add_eigenvector_aggregate_max_derivative(self, hb, rho, node):
        h_mag, eta, a, b = self.eigenvector_aggregate_magnitude(rho, node)

        eta_h = np.exp(rho * (h_mag - np.max(h_mag)))
        eta_h = eta_h / np.sum(eta_h)

        h = np.dot(eta_h, h_mag)

        def D(q):

            nn = len(q)
            eta_Dq = np.zeros(nn)

            for i in range(nn):
                eta_Dq[i] = eta_h[i] * q[i]
            return eta_Dq

        Qb = np.zeros(self.Q.shape)
        for i in range(self.N):
            Qb[node, i] += 2.0 * hb * eta[i] * D(self.Q[node, i])
            self.Qrb[:, i] += Qb[self.reduced, i]
            self.lamb[i] -= (
                hb
                * rho
                * eta[i]
                * (a[i] + b[i])
                * (self.Q[node, i].T @ D(self.Q[node, i]) - h)
            )

        return

    def get_compliance(self):
        fr = self.reduce_vector(self.f)

        # Compute the compliance
        compliance = 0.0
        for i in range(self.N):
            val = self.Qr[:, i].dot(fr)
            # compliance += (val * val) / self.lam[i]
            compliance += val * val

        return compliance

    def add_compliance_derivative(self, compb=1.0):
        fr = self.reduce_vector(self.f)

        for i in range(self.N):
            val = self.Qr[:, i].dot(fr)
            # self.Qrb[:, i] += 2.0 * compb * val * fr / self.lam[i]
            # self.lamb[i] -= compb * (val * val) / self.lam[i] ** 2
            self.Qrb[:, i] += 2.0 * compb * val * fr

        return

    def initialize(self, store=False):
        self.profile["total derivative time"] = 0.0
        self.profile["adjoint solution time"] = 0.0
        self.profile["matrix assembly time"] = 0.0
        self.profile["eigenvalue solve time"] = 0.0
        self.profile["solve preconditioner count"] = 0
        self.profile["adjoint preconditioner count"] = 0
        self.profile["koiter time"] = 0.0

        # Apply the filter
        self.rho = self.fltr.apply(self.x)

        # Average the density to get the element-wise density
        self.rhoE = np.mean(self.rho[self.conn[:, :4]], axis=1)

        # Solve the eigenvalue problem
        self.lam, self.Q = self.solve_eigenvalue_problem(store)

        # self.lam = self.lam[1:]
        # self.Q = self.Q[:, 1:]

        print("Eigenvalues: ", self.lam)
        if store:
            self.profile["eigenvalues"] = self.BLF.tolist()

        return

    def initialize_koiter(self, Q0_norm=None):
        self.l0 = self.lam[0]
        self.Q0 = self.Q[:, 0]

        self.a, self.b, self.u1, self.u2, self.K1, self.K11, self.G1 = (
            self.get_koiter_ab(self.rhoE, self.l0, self.u, self.Q0, Q0_norm)
        )

        return

    def initialize_koiter_derivative(self):
        t1 = time.time()

        self.dl0dx = self.get_dldx()

        # self.dK0dx, self.dG0dx, _ = self.get_dG0dx_dK0dx_du0dx_cs(
        #     self.x, self.K0, self.G0, self.u, self.Q0
        # )
        # self.dQ0dx, self.dl0dx = self.get_dQ0dx_dl0dx(
        #     self.x, self.K0, self.G0, self.dK0dx, self.dG0dx, self.l0, self.Q0
        # )
        # self.dadx = self.get_dadx_cd(self.x, self.K1, self.dQ0dx, self.Q0, dh)
        # self.dadx = self.get_dadx(self.rhoE, self.Q0, self.K1)

        self.dadx = self.get_dadx(self.rhoE, self.l0, self.Q0)
        self.dbdx = self.get_dbdx(self.rhoE, self.l0, self.Q0)
        t2 = time.time()
        self.profile["koiter time"] = t2 - t1
        ic(t2 - t1)

        return

    def initialize_adjoint(self):
        self.xb = np.zeros(self.x.shape)
        self.rhob = np.zeros(self.nnodes)

        self.lamb = np.zeros(self.lam.shape)
        self.Qrb = np.zeros(self.Qr.shape)

        return

    def check_adjoint_residual(self, A, B, lam, Q, Qb, psi, b_ortho=False):
        res, orth = eval_adjoint_residual_norm(A, B, lam, Q, Qb, psi, b_ortho=b_ortho)
        for i in range(Q.shape[1]):
            ratio = orth[i] / np.linalg.norm(Q[:, i])
            self.profile["adjoint norm[%2d]" % i] = res[i]
            self.profile["adjoint ortho[%2d]" % i] = ratio
            self.profile["adjoint lam[%2d]" % i] = lam[i]

        return res

    def add_check_adjoint_residual(self):
        return self.check_adjoint_residual(
            self.Gr, self.Kr, self.lam, self.Qr, self.Qrb, self.psir, b_ortho=False
        )

    def finalize_adjoint(self):

        class Callback:
            def __init__(self):
                self.res_list = []

            def __call__(self, rk=None):
                self.res_list.append(rk)

        callback = Callback()

        self.profile["adjoint solution method"] = self.adjoint_method
        self.factor.count = 0

        t0 = time.time()
        for i in range(self.cost):
            if i != 0:
                callback.res_list = []
            psir, corr_data = self.eig_solver.solve_adjoint(
                self.Qrb,
                rtol=self.rtol,
                method=self.adjoint_method,
                callback=callback,
                **self.adjoint_options,
            )
        t1 = time.time()
        t = (t1 - t0) / self.cost

        self.psir = psir

        self.profile["adjoint preconditioner count"] += self.factor.count
        self.profile["adjoint solution time"] += t
        self.profile["adjoint residuals"] = np.array(callback.res_list).tolist()
        self.profile["adjoint iterations"] = len(callback.res_list)
        logging.info("Adjoint solve time: %8.2f s" % t)
        self.profile["adjoint correction data"] = corr_data

        def dAdu(wr, vr):
            w = self.full_vector(wr)
            v = self.full_vector(vr)
            if w.ndim == 1 and v.ndim == 1:
                return self.get_stress_stiffness_matrix_uderiv(self.rhoE, w, v)

            elif w.ndim == 2 and v.ndim == 2:
                if self.dfds is None:
                    self.dfds = self.intital_stress_stiffness_matrix_deriv(
                        self.rhoE, self.Te, self.detJ, w, v
                    )
                return self.get_stress_stiffness_matrix_uderiv_tensor(
                    self.dfds, self.Be
                )

        dBdu = None

        # Compute the derivative of the function wrt the fundamental path
        # Compute the adjoint for K * adj = d ( psi^{T} * G(u, x) * phi ) / du
        dfdu0 = np.zeros(2 * self.nnodes)
        dfdu0 = self.eig_solver.add_total_derivative(
            self.lamb,
            self.Qrb,
            self.psir,
            dAdu,
            dBdu,
            dfdu0,
            adj_corr_data=corr_data,
            deriv_type=self.deriv_type,
        )

        # Create functions for computing dA/dx and dB/dx
        def dAdx(wr, vr):
            w = self.full_vector(wr)
            v = self.full_vector(vr)

            if w.ndim == 1 and v.ndim == 1:
                return self.get_stress_stiffness_matrix_xderiv(self.rhoE, self.u, w, v)

            elif w.ndim == 2 and v.ndim == 2:
                if self.dfds is None:
                    self.dfds = self.intital_stress_stiffness_matrix_deriv(
                        self.rhoE, self.Te, self.detJ, w, v
                    )
                return self.get_stress_stiffness_matrix_xderiv_tensor(
                    self.rhoE, self.u, self.dfds, self.Be
                )

        def dBdx(wr, vr):
            w = self.full_vector(wr)
            v = self.full_vector(vr)
            return self.get_stiffness_matrix_deriv(self.rhoE, w, v)

        self.rhob = self.eig_solver.add_total_derivative(
            self.lamb,
            self.Qrb,
            self.psir,
            dAdx,
            dBdx,
            self.rhob,
            adj_corr_data=corr_data,
            deriv_type=self.deriv_type,
        )

        # Solve the adjoint for the fundamental path
        dfdur = self.reduce_vector(dfdu0)
        psir = -self.Kfact(dfdur)
        psi = self.full_vector(psir)

        self.rhob += self.get_stiffness_matrix_deriv(self.rhoE, psi, self.u)
        self.xb += self.rhob

        t2 = time.time()
        self.profile["total derivative time"] += t2 - t1
        logging.info("Total derivative time: %5.2f s" % (t2 - t1))

        return

    def check_blf(self):
        """
        Check the BLF calculation using the eigensolver
        """

        self.initialize()
        print("\nResult BLF  = ", self.BLF)

        # Compute BLF use sparse.linalg.eigsh
        mu, _ = sparse.linalg.eigsh(
            self.Gr,
            M=self.Kr,
            k=self.N,
            sigma=self.sigma,
            which="SM",
            maxiter=1000,
            tol=1e-15,
        )
        BLF0 = -1.0 / mu[: self.N]

        # check if the BLF is correct
        print("Scipy.eigsh = ", BLF0)
        ic(np.allclose(BLF0, self.BLF))

    def check_buckling(self, psi, phi, dh=1e-6):
        # set random seed
        np.random.seed(1234)

        self.initialize()
        x = self.x
        Kr = self.Kr
        Gr = self.Gr
        Q0 = self.Q[:, 0]
        u1_norm = np.linalg.norm(Q0)
        u1 = Q0 / u1_norm

        # compute the full size eigenvalue problem
        mu, phi = eigh(Gr.todense(), Kr.todense())
        lam = -1.0 / mu
        ic(lam[0])

        # check if (Kr + lam * Gr) * phi = 0
        res = (Kr + lam[0] * Gr) @ phi[:, 0]
        ic(np.linalg.norm(res))

        # check if (Gr - mu * Kr) * phi = 0
        res = (Gr - mu[0] * Kr) @ phi[:, 0]
        ic(np.linalg.norm(res))

        dKdx = self.get_stiffness_matrix_deriv(self.rhoE, u1, u1)
        dGdx = self.get_stress_stiffness_matrix_xuderiv(self.rhoE, self.u, u1, u1)

        dh = 1e-4
        p = np.random.rand(self.x.size)
        # p = np.ones(self.x.size)
        p = p / np.linalg.norm(p)

        x1 = x + dh * p
        x2 = x - dh * p
        rho1 = self.fltr.apply(x1)
        rho2 = self.fltr.apply(x2)

        K1 = self.get_stiffness_matrix(rho1)
        K2 = self.get_stiffness_matrix(rho2)

        K1r = self.reduce_matrix(K1)
        K2r = self.reduce_matrix(K2)
        K1fact = linalg.factorized(K1r)
        K2fact = linalg.factorized(K2r)

        u1r = K1fact(self.fr)
        u2r = K2fact(self.fr)
        u01 = self.full_vector(u1r)
        u02 = self.full_vector(u2r)

        G1 = self.get_stress_stiffness_matrix(rho1, u01)
        G2 = self.get_stress_stiffness_matrix(rho2, u02)

        K_ans = p.T @ dKdx
        G_ans = p.T @ dGdx

        K_cd = u1.T @ (K1 - K2) @ u1 / (2 * dh)
        G_cd = u1.T @ (G1 - G2) @ u1 / (2 * dh)

        print("K Exact: ", K_ans)
        print("K CD   : ", K_cd)
        print("K Error: ", (K_ans - K_cd) / K_cd, "\n")

        print("G Exact: ", G_ans)
        print("G CD   : ", G_cd)
        print("G Error: ", (G_ans - G_cd) / G_cd, "\n")

        return

    def test_eigenvector_aggregate_derivatives(
        self, rho=100, dh_cd=1e-4, dh_cs=1e-20, node=None, pert=None, mode="tanh"
    ):

        hb = 1.0
        if node is None:
            node = (8 + 1) * 16 + 16

        # Initialize the problem
        self.initialize(store=True)

        # Copy the design variables
        x0 = np.array(self.x)

        # compute the compliance derivative
        self.initialize_adjoint()
        self.add_eigenvector_aggregate_derivative(hb, rho, node, mode=mode)
        self.finalize_adjoint()

        if pert is None:
            pert = np.random.uniform(size=self.x.shape)

        data = {}
        data["ans"] = np.dot(pert, self.xb)
        data.update(self.profile)

        if self.solver_type == "BasicLanczos":
            # Perturb the design variables for complex-step
            self.x = np.array(x0).astype(complex)
            self.x.imag += dh_cs * pert
            self.initialize()
            h1 = self.get_eigenvector_aggregate(rho, node, mode=mode)

            data["dh_cs"] = dh_cs
            data["cs"] = h1.imag / dh_cs
            data["cs_err"] = np.fabs((data["ans"] - data["cs"]) / data["cs"])

        self.x = x0 - dh_cd * pert
        self.initialize()
        h3 = self.get_eigenvector_aggregate(rho, node, mode=mode)

        self.x = x0 + dh_cd * pert
        self.initialize()
        h4 = self.get_eigenvector_aggregate(rho, node, mode=mode)

        data["dh_cd"] = dh_cd
        data["cd"] = (h4 - h3) / (2 * dh_cd)
        data["cd_err"] = np.fabs((data["ans"] - data["cd"]) / data["cd"])

        # Reset the design variables
        self.x = x0

        if self.solver_type == "BasicLanczos":
            print(
                "%25s  %25s  %25s  %25s  %25s"
                % ("Answer", "CS", "CD", "CS Rel Error", "CD Rel Error")
            )
            print(
                "%25.15e  %25.15e  %25.15e  %25.15e  %25.15e"
                % (data["ans"], data["cs"], data["cd"], data["cs_err"], data["cd_err"])
            )
        else:
            print("%25s  %25s  %25s" % ("Answer", "CD", "CD Rel Error"))
            print(
                "%25.15e  %25.15e  %25.15e" % (data["ans"], data["cd"], data["cd_err"])
            )

        return data

    def test_compliance_derivatives(self, dh_fd=1e-4, pert=None):
        # Initialize the problem
        self.initialize(store=True)

        # Copy the design variables
        x0 = np.array(self.x)

        # compute the compliance derivative
        self.initialize_adjoint()
        self.add_compliance_derivative()
        self.finalize_adjoint()

        pert = np.random.uniform(size=x0.shape)

        ans = np.dot(pert, self.xb)

        self.x = x0 + dh_fd * pert
        self.initialize()
        c1 = self.get_compliance()

        self.x = x0 - dh_fd * pert
        self.initialize()
        c2 = self.get_compliance()

        cd = (c1 - c2) / (2 * dh_fd)

        print("\nTotal derivative for compliance")
        print("Ans = ", ans)
        print("CD  = ", cd)
        print("Rel err = ", (ans - cd) / cd, "\n")

        return

    def test_ks_buckling_derivatives(self, dh_fd=1e-4, ks_rho=160, pert=None):
        # Initialize the problem
        self.initialize(store=True)

        # Copy the design variables
        x0 = np.array(self.x)

        # compute the compliance derivative
        t0 = time.time()
        dks = self.eval_ks_buckling_derivative(ks_rho)
        t1 = time.time()

        pert = np.random.uniform(size=x0.shape)

        ans = np.dot(pert, dks)

        self.x = x0 + dh_fd * pert
        self.initialize()
        c1 = self.eval_ks_buckling(ks_rho)

        self.x = x0 - dh_fd * pert
        self.initialize()
        c2 = self.eval_ks_buckling(ks_rho)

        cd = (c1 - c2) / (2 * dh_fd)

        self.x = x0
        self.initialize()

        print("\nTotal derivative for ks-buckling:", self.deriv_type + " type")
        print("Ans:                 ", ans)
        print("CD:                  ", cd)
        print("Rel err:             ", (ans - cd) / cd)
        print("Time for derivative: ", t1 - t0, "s")

        return

    def test_true_compliance_derivatives(self, dh_fd=1e-4, pert=None):
        # Initialize the problem
        self.initialize(store=True)

        # Copy the design variables
        x0 = np.array(self.x)

        # compute the compliance derivative
        dks = self.compliance_derivative()

        pert = np.random.uniform(size=x0.shape)

        ans = np.dot(pert, dks)

        self.x = x0 + dh_fd * pert
        self.initialize()
        c1 = self.compliance()

        self.x = x0 - dh_fd * pert
        self.initialize()
        c2 = self.compliance()

        cd = (c1 - c2) / (2 * dh_fd)

        print("\nTotal derivative for true compliance")
        print("Ans:                 ", ans)
        print("CD:                  ", cd)
        print("Rel err:             ", (ans - cd) / cd)

        return

    def test_eigenvector_aggregate_max_derivatives(
        self, dh_fd=1e-4, rho_agg=100, pert=None, node=None
    ):

        hb = 1.0
        if node is None:
            node = []
            for i in range(self.nnodes):
                node.append(i)

        # Initialize the problem
        self.initialize(store=True)

        # Copy the design variables
        x0 = np.array(self.x)

        # compute the compliance derivative
        self.initialize_adjoint()
        self.add_eigenvector_aggregate_max_derivative(hb, rho_agg, node)
        self.finalize_adjoint()

        pert = np.random.uniform(size=x0.shape)

        ans = np.dot(pert, self.xb)

        self.x = x0 + dh_fd * pert
        self.initialize()
        h1 = self.get_eigenvector_aggregate_max(rho_agg, node)

        self.x = x0 - dh_fd * pert
        self.initialize()
        h2 = self.get_eigenvector_aggregate_max(rho_agg, node)
        cd = (h1 - h2) / (2 * dh_fd)

        print("\nTotal derivative for aggregate-max")
        print("Ans = ", ans)
        print("CD  = ", cd)
        print("Rel err = ", (ans - cd) / cd, "\n")

        return

    def get_pts_and_tris(self, eta=None):
        pts = np.zeros((self.nnodes, 3))

        if eta is not None:
            u = self.Q.dot(eta)
            pts[:, 0] = self.X[:, 0] + 10 * u[::2]
            pts[:, 1] = self.X[:, 1] + 10 * u[1::2]

        # Create the triangles
        tris = np.zeros((2 * self.nelems, 3), dtype=int)
        tris[: self.nelems, 0] = self.conn[:, 0]
        tris[: self.nelems, 1] = self.conn[:, 1]
        tris[: self.nelems, 2] = self.conn[:, 2]

        tris[self.nelems :, 0] = self.conn[:, 0]
        tris[self.nelems :, 1] = self.conn[:, 2]
        tris[self.nelems :, 2] = self.conn[:, 3]

        return pts, tris, self.rho

    def plot(self, field, u=None, scale=1.0, ax=None, **kwargs):
        """
        Create a plot
        """

        # Create the triangles
        triangles = np.zeros((2 * self.nelems, 3), dtype=int)
        triangles[: self.nelems, 0] = self.conn[:, 0]
        triangles[: self.nelems, 1] = self.conn[:, 1]
        triangles[: self.nelems, 2] = self.conn[:, 2]

        triangles[self.nelems :, 0] = self.conn[:, 0]
        triangles[self.nelems :, 1] = self.conn[:, 2]
        triangles[self.nelems :, 2] = self.conn[:, 3]

        # Create the triangulation object
        if u is None:
            x = self.X[:, 0]
            y = self.X[:, 1]
        else:
            x = self.X[:, 0] + scale * u[0::2]
            y = self.X[:, 1] + scale * u[1::2]
        tri_obj = tri.Triangulation(x, y, triangles)

        if ax is None:
            fig, ax = plt.subplots()

        # Set the aspect ratio equal
        ax.set_aspect("equal")

        # Create the contour plot
        ax.tricontourf(tri_obj, field, **kwargs)

        return

    def plot_design(self, path=None, index=None):
        fig, ax = plt.subplots()
        self.plot(self.rho, ax=ax)

        # # plot the nodes
        # for i in range(self.nnodes):
        #     ax.scatter(self.X[i, 0], self.X[i, 1], color="k", s=1, clip_on=False)

        # # plot the conn
        # for i in range(self.nelems):
        #     for j in range(4):
        #         x0 = self.X[self.conn[i, j], 0]
        #         y0 = self.X[self.conn[i, j], 1]
        #         x1 = self.X[self.conn[i, (j + 1) % 4], 0]
        #         y1 = self.X[self.conn[i, (j + 1) % 4], 1]
        #         ax.plot([x0, x1], [y0, y1], color="k", linewidth=0.5, clip_on=False)

        # # plot the bcs
        # for i, v in self.bcs.items():
        #     ax.scatter(self.X[i, 0], self.X[i, 1], color="b", s=5, clip_on=False)

        # plot the non-design nodes
        # m0_X = np.array([self.X[i, :] for i in self.fltr.non_design_nodes])
        # ax.scatter(m0_X[:, 0], m0_X[:, 1], color="blue", s=5, clip_on=False)

        # for i, v in self.forces.items():
        #     ax.quiver(
        #         self.X[i, 0],
        #         self.X[i, 1],
        #         v[0],
        #         v[1],
        #         color="r",
        #         scale=1e-3,
        #         clip_on=False,
        #     )

        # if index is not None:
        #     for i in index:
        #         ax.scatter(
        #             self.X[i, 0], self.X[i, 1], color="orange", s=5, clip_on=False
        #         )

        ## add midline vertical line
        # ax.plot([0.5, 0.5], [0, 2], color="k", linestyle="--")

        ## add midline horizontal line
        # ax.plot([0, 1], [1, 1], color="k", linestyle="--")

        ax.set_aspect("equal")
        ax.axis("off")

        if path is not None:
            fig.savefig(path, bbox_inches="tight", dpi=150)

        plt.close(fig)

        return

    def plot_topology(self, ax):
        # Set the number of levels to use.
        levels = np.linspace(0.0, 1.0, 26)

        # Make sure that there are no ticks on either axis (these affect the bounding-box
        # and make extra white-space at the corners). Finally, turn off the axis so its
        # not visible.
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.axis("off")

        self.plot(self.rho, ax=ax, levels=levels, cmap="viridis", extend="max")

        return

    def plot_mode(self, k, ax):
        if k < self.N and k >= 0 and self.Q is not None:
            # Set the number of levels to use.
            levels = np.linspace(0.0, 1.0, 26)

            # Make sure that there are no ticks on either axis (these affect the bounding-box
            # and make extra white-space at the corners). Finally, turn off the axis so its
            # not visible.
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.axis("off")

            value = np.fabs(np.max(self.Q[:, k])) + np.fabs(np.min(self.Q[:, k]))
            scale = 0.5 / value

            self.plot(
                self.rho,
                ax=ax,
                u=self.Q[:, k],
                scale=scale,
                levels=levels,
                cmap="viridis",
                extend="max",
            )

        return

    def plot_residuals(self, path=None):
        fig, ax = plt.subplots()
        ax.plot(self.profile["adjoint residuals"], marker="o")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual")

        if path is not None:
            fig.savefig(path, bbox_inches="tight", dpi=300)

        plt.close(fig)
        return fig, ax

    def update(self, i, u, l, ax, title):
        levels = np.linspace(0.0, 1.0, 26)
        ax.clear()
        self.plot(
            self.rho,
            u=u[i, :],
            ax=ax,
            levels=levels,
            cmap="Greys",
            extend="max",
        )
        ax.set_aspect("equal")
        ax.axis("off")

        for j, v in self.forces.items():
            ax.quiver(
                self.X[j, 0] + u[i, j * 2],
                self.X[j, 1] + u[i, j * 2 + 1],
                v[0],
                v[1],
                color="r",
                scale=1e-3,
                clip_on=False,
                pivot="tip",
            )

        ax.set_title(title + r": $\lambda$" + f"= %.2f" % l[i])
        return

    def video_u(self, u, l, path, title=""):
        fig, ax = plt.subplots()
        update = lambda i: self.update(i, u, l, ax, title)
        anim = FuncAnimation(fig, update, frames=len(l), interval=100)
        anim.save(path, dpi=300)
        return ax

    def plot_u(self, u, l, eigvals, path=None):
        # get the middle node of the index where the force is applied
        indy = 2 * np.nonzero(self.f[1::2])[0] + 1
        indy = indy[len(indy) // 2]
        indx = indy - 1

        # compute the index where eigvals are negative and positive
        indx_neg = np.where(eigvals < 0)[0]
        indx_pos = np.where(eigvals > 0)[0]

        fig, ax = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
        ax[0].plot(u[:, indx], l / self.lam[0], marker="o")
        ax[0].set_xlabel(r"$u_x$")
        ax[0].set_ylabel(r"$\lambda/\lambda_c$")
        ax[0].axvline(x=0, color="grey", linestyle="--")
        ax[0].axhline(y=1, color="grey", linestyle="--")

        # add twin y axis to plot eigvals
        ax2 = ax[0].twinx()
        ax2.semilogy(u[:, indx], np.abs(eigvals), color="k")
        ax2.scatter(u[indx_neg, indx], np.abs(eigvals[indx_neg]), color="b", zorder=10)
        ax2.scatter(u[indx_pos, indx], eigvals[indx_pos], color="r", zorder=10)
        ax2.set_ylabel(r"$|\lambda_0(K_T)|$", color="r")

        ax[1].plot(u[:, indy], l / self.lam[0], marker="o")
        ax[1].set_xlabel(r"$u_y$")
        ax[1].set_ylabel(r"$\lambda/\lambda_c$")
        ax[1].axvline(x=0, color="k", linestyle="--")
        ax[1].axhline(y=1, color="k", linestyle="--")

        ax2 = ax[1].twinx()
        ax2.semilogy(u[:, indy], np.abs(eigvals), color="k")
        ax2.scatter(u[indx_neg, indy], np.abs(eigvals[indx_neg]), color="b", zorder=10)
        ax2.scatter(u[indx_pos, indy], eigvals[indx_pos], color="r", zorder=10)
        ax2.set_ylabel(r"$|\lambda_0(K_T)|$", color="r")
        ax[1].invert_xaxis()

        plt.savefig(path, bbox_inches="tight")

        return ax

    def plot_u_compare(self, a, b, u1, u2, path=None, lam_s=None):
        # get the middle node of the index where the force is applied
        indy = 2 * np.nonzero(self.f[1::2])[0] + 1
        indy = indy[len(indy) // 2]
        indx = indy - 1

        fig, ax = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)

        xib0 = [0, 0.0001, 0.001, 0.01]

        cb = [cw(0.0), cw(0.1), cw(0.2), cw(0.3)]
        cr = [cw(1.0), cw(0.9), cw(0.8), cw(0.7)]

        # scale = [0.4, 0.4, 0.95, 6]
        scale = [1.8, 1.8, 1.8, 2.2]
        u_arc = []
        l_arc = []

        # for i in range(len(xib0)):
        #     u, l = self.arc_length_method(
        #         Dl=self.lam[0] * 0.2, # 0.2
        #         lmax=self.lam[0] * 5,
        #         scale=scale[i],
        #         u_imp=xib0[i] * self.Q[:, 0],
        #     )
        #     u_arc.append(u)
        #     l_arc.append(l)

        # np.save("./output/u_arc.npy", u_arc)
        # np.save("./output/l_arc.npy", l_arc)

        # u_arc = np.load("./output/u_arc.npy")
        # l_arc = np.load("./output/l_arc.npy")

        # for i in range(len(xib0)):
        #     if xib0[i] == 0:
        #         label = r"Arc-Length $\bar{\xi}$ = 0"
        #     else:
        #         label = r"Arc-Length $\bar{\xi}$ = %.e$\phi_1$" % np.abs(xib0[i])

        #     ax[0].plot(
        #         u_arc[i][:, indx],
        #         l_arc[i] / self.lam[0],
        #         marker="o",
        #         label=label,
        #         color=cb[i],
        #     )
        #     ax[1].plot(
        #         u_arc[i][:, indy],
        #         l_arc[i] / self.lam[0],
        #         marker="o",
        #         label=label,
        #         color=cb[i],
        #     )

        # apply the Koiter-Asymptotic
        xi = np.linspace(0, -5e1, 100)

        # generate the Koiter-Asymptotic for different imperfection
        u = []
        lam = []
        for i in range(len(xib0)):
            xib = xib0[i] * np.linalg.norm(topo.Q[:, 0])

            # compute lambda based on xi value
            if xib == 0:
                lam_imp = (1 + a * xi + b * xi**2) * self.lam[0]
            else:
                lam_imp = (xi + a * xi**2 + b * xi**3) / (xi + xib) * self.lam[0]

            # compute the displacement based on lambda and xi
            u_imp = np.zeros((lam_imp.size, topo.nvars))
            for n in range(lam_imp.size):
                u_imp[n, :] = lam_imp[n] * self.u + xi[n] * u1 + xi[n] ** 2 * u2

            if xib == 0:
                u_imp = np.vstack((np.zeros(topo.nvars), u_imp))
                lam_imp = np.hstack((0, lam_imp))

            # bb = -4 * a * xib
            # cc = 4 * a * xib
            # f = 0.5 * (-bb + np.sqrt(bb**2 - 4 * cc))
            # lam_s = (1 - f) * self.lam[0]
            axi = np.abs(a * xib)
            lam_s = self.lam[0] * (1 + 2 * axi - 2 * np.sqrt(axi + axi**2))
            ic(lam_s)

            if xib0[i] == 0:
                la = r"Koiter-Asymptotic $\bar{\xi}$ = 0"
            else:
                la = r"Koiter-Asymptotic $\bar{\xi}$ = %.e$\phi_1$" % np.abs(xib0[i])
            l0 = ax[0].plot(
                u_imp[:, indx],
                lam_imp / self.lam[0],
                marker="o",
                label=la,
                ms=2,
                color=cr[i],
            )
            l2 = ax[1].plot(
                u_imp[:, indy],
                lam_imp / self.lam[0],
                marker="o",
                label=la,
                ms=2,
                color=cr[i],
            )

            ax[0].axhline(
                y=lam_s / self.lam[0], color=l0[0].get_color(), linestyle="--"
            )
            ax[1].axhline(
                y=lam_s / self.lam[0], color=l2[0].get_color(), linestyle="--"
            )

            u.append(u_imp)
            lam.append(lam_imp)

        ax[0].set_xlabel(r"$u_x$")
        ax[0].set_ylabel(r"$\lambda/\lambda_c$")
        # ax[0].axvline(x=0, color="grey", linestyle="--")
        # ax[0].axhline(y=1, color="grey", linestyle="--")
        ax[0].legend()

        ax[1].set_xlabel(r"$u_y$")
        ax[1].set_ylabel(r"$\lambda/\lambda_c$")
        # ax[1].axvline(x=0, color="grey", linestyle="--")
        # ax[1].axhline(y=1, color="grey", linestyle="--")
        ax[1].invert_xaxis()
        ax[1].legend()

        plt.savefig(path, bbox_inches="tight")

        return u, lam, xib0


def domain_compressed_column(nx=64, ny=128, Lx=1.0, Ly=2.0, shear_force=False):
    """
    ________
    |      |
    |      |
    |      | ny
    |      |
    |______|
       nx
    """
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)

    nelems = nx * ny
    nnodes = (nx + 1) * (ny + 1)
    nodes = np.arange(nnodes, dtype=int).reshape(nx + 1, ny + 1)
    conn = np.zeros((nelems, 4), dtype=int)
    X = np.zeros((nnodes, 2))

    for j in range(ny + 1):
        for i in range(nx + 1):
            X[nodes[i, j], 0] = x[i]
            X[nodes[i, j], 1] = y[j]

    for j in range(ny):
        for i in range(nx):
            conn[i + nx * j, 0] = nodes[i, j]
            conn[i + nx * j, 1] = nodes[i + 1, j]
            conn[i + nx * j, 2] = nodes[i + 1, j + 1]
            conn[i + nx * j, 3] = nodes[i, j + 1]

    # Create the symmetry in the problem
    dvmap = np.zeros((nx + 1, ny + 1), dtype=int)
    index = 0

    # 2-way reflection left to right
    for i in range(nx // 2 + 1):
        for j in range(ny + 1):
            if dvmap[i, j] >= 0:
                dvmap[i, j] = index
                dvmap[nx - i, j] = index
                index += 1

    num_design_vars = index
    dvmap = dvmap.flatten()

    # apply boundary conditions at the bottom nodes
    bcs = {}
    for i in range(nx + 1):
        bcs[nodes[i, 0]] = [0, 1]

    # apply a force at the top middle
    # force is independent of the mesh size,
    P = 1e-3
    forces = {}

    if shear_force is True:
        # apply a shear force at the top middle
        for i in range(nx + 1):
            forces[nodes[i, ny]] = [P / (nx + 1), 0]

    else:
        # apply a vertical force at the top middle
        offset = int(np.ceil(nx / 30))
        for i in range(offset):
            forces[nodes[nx // 2 - i - 1, ny]] = [0, -P / (2 * offset + 1)]
            forces[nodes[nx // 2 + i + 1, ny]] = [0, -P / (2 * offset + 1)]
        forces[nodes[nx // 2, ny]] = [0, -P / (2 * offset + 1)]

    non_design_nodes = []

    ic(num_design_vars, len(non_design_nodes))

    return conn, X, dvmap, num_design_vars, non_design_nodes, bcs, forces


def domain_rooda_frame(nx=120, l=8.0, lfrac=0.1):
    """
     _____nx_____  _______________
    |            |               ^
    |      ______|               |
    |     |                     l
    |     |       lfrac * l     |
    |__nt_|        _____________|

    """
    nt = int(np.ceil(nx * lfrac))
    # check if the number of elements is odd
    if nt % 2 != 0:
        nt += 1

    nelems = nx * nx - (nx - nt) * (nx - nt)
    nnodes = (nx + 1) * (nx + 1) - (nx - nt) * (nx - nt)

    nodes_1 = np.arange((nx - nt) * (nt + 1)).reshape(nx - nt, nt + 1)
    nodes_2 = (nx - nt) * (nt + 1) + np.arange((nt + 1) * (nx + 1)).reshape(
        nt + 1, nx + 1
    )

    def ij_to_node(ip, jp):
        if jp < nx - nt:
            return nodes_1[jp, ip]
        return nodes_2[jp - (nx - nt), ip]

    def pt_out_domain(ip, jp):
        return ip > nt and jp < nx - nt

    def elem_out_domain(ie, je):
        return ie >= nt and je <= nx - nt - 1

    X = np.zeros((nnodes, 2))
    index = 0
    for jp in range(nx + 1):  # y-directional index
        for ip in range(nx + 1):  # x-directional index
            if not pt_out_domain(ip, jp):
                X[index, :] = [l / nx * ip, l / nx * jp]
                index += 1

    conn = np.zeros((nelems, 4), dtype=int)
    index = 0
    for je in range(nx):  # y-directional index
        for ie in range(nx):  # x-directional index
            if not elem_out_domain(ie, je):
                conn[index, :] = [
                    ij_to_node(ie, je),
                    ij_to_node(ie + 1, je),
                    ij_to_node(ie + 1, je + 1),
                    ij_to_node(ie, je + 1),
                ]
                index += 1

    t = 1
    bcs = {}
    non_design_nodes = []

    # bottom middle ___ fixed
    bcs[ij_to_node((nt + 1) // 2, 0)] = [0, 1]

    # top middle | fixed
    bcs[ij_to_node(nx, nx - nt // 2)] = [0, 1]

    # # bottom left corner ___ fixed
    # for ip in range(nt + 1):
    #     bcs[ij_to_node(ip, 0)] = [0, 1]

    # # top right corner | fixed
    # for jp in range(nx - nt, nx + 1):
    #     bcs[ij_to_node(nx, jp)] = [0, 1]

    # for jp in range(nx + 1):
    #     for aa in range(t):
    #         non_design_nodes.append(ij_to_node(aa, jp))

    # for jp in range(nx - nt):
    #     for aa in range(t):
    #         non_design_nodes.append(ij_to_node(nt-aa, jp))

    # for ip in range(t, nx + 1):
    #     for aa in range(t):
    #         non_design_nodes.append(ij_to_node(ip, nx-aa))

    # for ip in range(nt + 1 - t, nx + 1):
    #     for aa in range(t):
    #         non_design_nodes.append(ij_to_node(ip, nx-nt + aa))

    forces = {}
    P = -5e-4
    P_nnodes = int(np.ceil(0.01 * nx))
    P_pst = int(np.ceil(0.2 * nx))

    P_pst = 0
    P_nnodes = nt + 1

    P_pst = (nt + 1) // 2
    P_nnodes = 1

    for ip in range(P_pst, P_pst + P_nnodes):
        forces[ij_to_node(ip, nx)] = [0, P / P_nnodes]

    # remove the non-design nodes
    num_design_vars = nnodes  # - len(non_design_nodes)

    # dvmap = np.arange(nnodes)
    # dvmap[non_design_nodes] = -1
    dvmap = None

    ic(num_design_vars, len(non_design_nodes))

    return conn, X, dvmap, num_design_vars, non_design_nodes, bcs, forces


def make_model(nx=64, ny=128, Ly=2.0, rfact=4.0, N=10, shear_force=False, **kwargs):
    """

    Parameters
    ----------
    ny : int
        Number of nodes in the y-direction
    rfact : real
        Filter radius as a function of the element side length
    N : int
        Number of eigenvalues and eigenvectors to compute
    """

    if kwargs.get("domain") == "column":
        ny = int(Ly * nx)
        r0 = rfact * (1.0 / nx)
        conn, X, dvmap, num_design_vars, non_design_nodes, bcs, forces = (
            domain_compressed_column(
                nx=nx, ny=ny, Lx=1.0, Ly=Ly, shear_force=shear_force
            )
        )
    elif kwargs.get("domain") == "rooda":
        r0 = rfact * (8.0 / nx)
        conn, X, dvmap, num_design_vars, non_design_nodes, bcs, forces = (
            domain_rooda_frame(nx=nx, l=8.0, lfrac=0.1)
        )
    else:
        raise ValueError("Invalid domain")

    fltr = NodeFilter(
        conn,
        X,
        r0=r0,
        dvmap=dvmap,
        num_design_vars=num_design_vars,
        non_design_nodes=non_design_nodes,
        projection=kwargs.get("projection"),
        beta=kwargs.get("b0"),
    )

    # delete the projection and beta from the kwargs
    if "projection" in kwargs:
        del kwargs["projection"]
    if "b0" in kwargs:
        del kwargs["b0"]
    if "domain" in kwargs:
        del kwargs["domain"]

    topo = TopologyAnalysis(fltr, conn, X, bcs=bcs, forces=forces, N=N, **kwargs)

    return topo


if __name__ == "__main__":
    import sys

    kokkos.initialize()

    # enable logging
    logging.getLogger("matplotlib.font_manager").disabled = True

    # Eliminate the randomness
    np.random.seed(0)

    if "ad-adjoint" in sys.argv:
        method = "ad-adjoint"
        adjoint_options = {"lanczos_guess": False}
    elif "pcpg" in sys.argv:
        method = "pcpg"
        adjoint_options = {"lanczos_guess": True}
    elif "gmres" in sys.argv:
        method = "gmres"
        adjoint_options = {"lanczos_guess": True}
    elif "approx-lanczos" in sys.argv:
        method = "approx-lanczos"
        adjoint_options = {}
    else:
        method = "shift-invert"
        adjoint_options = {
            "lanczos_guess": True,
            "update_guess": True,
            "bs_target": 1,
        }

    solver_type = "IRAM"
    if "IRAM" in sys.argv:
        solver_type = "IRAM"

    print("method = ", method)
    print("adjoint_options = ", adjoint_options)
    print("solver_type = ", solver_type)

    topo = make_model(
        nx=300,  # 32 x 64 mesh
        rfact=3.0,
        N=1,
        sigma=15,
        solver_type=solver_type,
        adjoint_method=method,
        adjoint_options=adjoint_options,
        shear_force=False,  # True,
        deriv_type="vector",
        domain="rooda",  # "column", "rooda",
        b0=16.0,
        p=6.0,
        projection=True,
    )

    # check the buckling load factor vs sparse.linalg.eigsh
    # topo.check_blf()

    # # check derivatives of the K, G matrices
    # q1 = np.random.RandomState(1).rand(topo.nvars)
    # q2 = np.random.RandomState(2).rand(topo.nvars)
    # topo.check_buckling(q1, q2)

    # check the compliance derivatives
    # topo.test_compliance_derivatives()

    # check the ks-buckling derivatives
    # topo.test_ks_buckling_derivatives()
    # topo.test_true_compliance_derivatives()
    # topo.test_eigenvector_aggregate_max_derivatives()

    # # check the eigenvector aggregate derivatives
    # data = topo.test_eigenvector_aggregate_derivatives(mode="tanh", rho=1000.0)
    # print("adjoint solution time", data["adjoint solution time"])
    # print("total derivative time", data["total derivative time"])

    topo.initialize()
    # topo.testR()
    # topo.testKt()

    # topo.check_derivative(topo.rhoE, topo.lam[0], topo.u, topo.Q[:, 0])

    # exit()

    # lam0 = 1.0
    # u0 = topo.newton_raphson(lam=lam0)
    # u0, u1, u2, u3 = topo.compute_path(topo.rhoE, f=topo.f, u0=u0)

    # lam = np.linspace(lam0, 1.5, 10)
    # s = lam - lam0
    # # ue = u0 + s * u1 + 0.5 * s**2 * u2 + (1 / 6) * s**3 * u3
    # ue = np.zeros((lam.size, topo.nvars))
    # for i in range(lam.size):
    #     ue[i, :] = u0 + s[i] * u1 + 0.5 * s[i] ** 2 * u2 + (1 / 6) * s[i] ** 3 * u3

    # topo.video_u(ue, lam, "buckling_path.mp4")
    # # topo.plot_u(ue, lam, np.zeros(lam.size), "buckling_path.pdf")
    # u1 = np.load("./output/u_list.npy")
    # l1 = np.load("./output/lam_list.npy")

    # topo.plot_u_compare(u1, l1, ue, lam, "buckling_path.pdf")

    # # topo.plot_design("design.pdf")
    # lmax = topo.lam[0] * 5

    # u_list, lam_list, eigvals = topo.arc_length_method(
    #     Dl=topo.lam[0] * 0.2,
    #     scale=16,
    #     lmax=topo.lam[0] * 5,
    #     geteigval=False,
    #     maxiter=100,
    #     u_imp=-1e-2 * topo.Q[:, 0],
    # )
    # np.save("./output/u_list.npy", u_list)
    # np.save("./output/lam_list.npy", lam_list)
    # np.save("./output/eigvals.npy", eigvals)

    # u_newton = topo.newton_raphson(lam=max(lam_list))
    # np.save("./output/u_newton.npy", u_newton)

    # u_list = np.load("./output/u_list.npy")
    # lam_list = np.load("./output/lam_list.npy")
    # eigvals = np.load("./output/eigvals.npy")
    # u_newton = np.load("./output/u_newton.npy")

    # plot the u as a video that shows the buckling process
    # topo.video_u(u_list, lam_list, "arc_length.mp4", "Arc-Length")
    # topo.plot_u(u_list, lam_list, eigvals, "arc_length.pdf")

    # u1, _= topo.arc_length_method(Dl=topo.lam[0] * 0.1, lmax=lam_c)

    # a, b, u1, u2 = topo.get_koiter_ab(topo.rhoE, lam_c, u1, topo.Q[:, 0])
    # u0 = topo.u / np.linalg.norm(topo.u)
    # lam_c = 1.71
    # u1 = topo.newton_raphson(lam=lam_c)

    topo.initialize_koiter()
    a = topo.a
    b = topo.b
    u1 = topo.u1
    u2 = topo.u2

    print("a = ", a)
    print("b = ", b)

    # generate the Koiter-Asymptotic comparison plot
    u, lam, xib0 = topo.plot_u_compare(a, b, u1, u2, "comparsion.pdf")

    # # generate the Koiter-Asymptotic video
    # for i in range(len(u)):
    #     ti = r"Koiter-Asymptotic $\bar{\xi}$ = %.e$\phi_1$" % np.abs(xib0[i])
    #     topo.video_u(u[i], lam[i], r"koiter_imp_%.e.mp4" % np.abs(xib0[i]), title=ti)

    # fig, ax = plt.subplots(1, 5, figsize=(15, 5))
    # levels = np.linspace(0.0, 1.0, 26)
    # uu = [
    #     u_list[lam_list.size // 4 * 1, :],
    #     u_list[lam_list.size // 4 * 2, :],
    #     u_list[lam_list.size // 4 * 3, :],
    #     u_list[-1, :],
    #     u_newton,
    # ]
    # for i in range(5):
    #     ii = lam_list.size // 4 * (i + 1)
    #     topo.plot(
    #         topo.rho,
    #         u=uu[i],
    #         ax=ax[i],
    #         levels=levels,
    #         cmap="Greys",
    #         extend="max",
    #     )
    #     if i < 3:
    #         ax[i].title.set_text(r"Arc-Length $\lambda$" + f"= %.2f" % lam_list[ii])
    #     elif i == 3:
    #         ax[i].title.set_text(r"Arc-Length $\lambda$" + f"= %.2f" % lam_list[-1])
    #     else:
    #         ax[i].title.set_text(r"Newton-Raphson $\lambda$" + f"= %.2f" % lam_list[-1])

    #     ax[i].set_aspect("equal")
    #     ax[i].set_xticks([])
    #     ax[i].set_yticks([])

    # plt.savefig("u_history.pdf", bbox_inches="tight")

    kokkos.finalize()

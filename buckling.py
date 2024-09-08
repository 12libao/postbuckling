import json
import logging
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

import eigenvector_derivatives as eig_deriv
from eigenvector_derivatives import (
    IRAM,
    BasicLanczos,
    SpLuOperator,
    eval_adjoint_residual_norm,
)
from icecream import ic
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
        rho0_G=1e-9,
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

        print("nnodes", self.nnodes)

        # # Set the initial design variable values
        # def _read_vtk(file, nx, ny):
        #     x, rho = [], []
        #     nnodes = int((nx + 1) * (ny + 1))
        #     ic(nx, ny, nnodes)
        #     with open(file) as f:
        #         for num, line in enumerate(f, 1):
        #             if "design" in line:
        #                 x = np.loadtxt(file, skiprows=num + 1, max_rows=nnodes)
        #             if "rho" in line:
        #                 rho = np.loadtxt(file, skiprows=num + 1, max_rows=nnodes)
        #                 break

        #     rho = rho.reshape((nx + 1, ny + 1))
        #     x = x.reshape((nx + 1, ny + 1))

        #     return x, rho

        # self.x, self.rho = _read_vtk("./output/it_1000.vtk", 200, 400)

        # self.x = self.x.flatten()
        # self.rho = self.rho.flatten()

        self.x = np.ones(self.fltr.num_design_vars)
        self.rho = self.fltr.apply(self.x)
        self.xb = np.zeros(self.x.shape)

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

    def get_stiffness_matrix(self, rhoE):
        """
        Assemble the stiffness matrix
        """

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
            Ke += detJ[:, np.newaxis, np.newaxis] * Be.transpose(0, 2, 1) @ C @ Be

        K = sparse.coo_matrix((Ke.flatten(), (self.i, self.j)))
        K = K.tocsr()

        return K

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
                # se = np.einsum("nij,nj -> ni", Be, psie)
                # te = np.einsum("nij,nj -> ni", Be, ue)
                # dfdrhoE += detJ * np.einsum("ij,nj,ni -> n", self.C0, se, te)

                for n in range(self.nelems):
                    se = Be[n] @ psie[n]
                    te = Be[n] @ ue[n]
                    # dfdrhoE[n] += detJ[n] * np.dot(se, np.dot(self.C0, te))
                    dfdrhoE[n] += detJ[n] * se.T @ self.C0 @ te
                    # dfdrhoE[n] += detJ[n] * psie[n].T @ Be[n].T @ self.C0 @ Be[n] @ ue[n]

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

        return dKdrho

    def get_mass_matrix(self, rhoE):
        """
        Assemble the mass matrix
        """

        # Compute the element density
        if self.ptype_M == "msimp":
            nonlin = self.simp_c1 * rhoE**6.0 + self.simp_c2 * rhoE**7.0
            cond = (rhoE > 0.1).astype(int)
            density = self.density * (rhoE * cond + nonlin * (1 - cond))
        elif self.ptype_M == "ramp":
            density = self.density * (
                (self.q + 1.0) * rhoE / (1 + self.q * rhoE) + self.rho0_M
            )
        else:  # linear
            density = self.density * rhoE

        # Assemble all of the the 8 x 8 element mass matrices
        Me = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)

        for j in range(2):
            for i in range(2):
                detJ = self.detJ[:, 2 * j + i]
                He = self.He[:, :, 2 * j + i]

                # This is a fancy (and fast) way to compute the element matrices
                Me += np.einsum("n,nij,nil -> njl", density * detJ, He, He)

        M = sparse.coo_matrix((Me.flatten(), (self.i, self.j)))
        M = M.tocsr()

        return M

    def get_mass_matrix_deriv(self, rhoE, u, v):
        """
        Compute the derivative of the mass matrix
        """

        # Derivative with respect to element density
        dfdrhoE = np.zeros(self.nelems)

        # The element-wise variables
        ue = np.zeros((self.nelems, 8) + u.shape[1:])
        ve = np.zeros((self.nelems, 8) + v.shape[1:])

        ue[:, ::2, ...] = u[2 * self.conn, ...]
        ue[:, 1::2, ...] = u[2 * self.conn + 1, ...]

        ve[:, ::2, ...] = v[2 * self.conn, ...]
        ve[:, 1::2, ...] = v[2 * self.conn + 1, ...]

        if u.ndim == 1 and v.ndim == 1:
            for j in range(2):
                for i in range(2):
                    He = self.He[:, :, 2 * j + i]
                    detJ = self.detJ[:, 2 * j + i]

                    eu = np.einsum("nij,nj -> ni", He, ue)
                    ev = np.einsum("nij,nj -> ni", He, ve)
                    dfdrhoE += np.einsum("n,ni,ni -> n", detJ, eu, ev)
        elif u.ndim == 2 and v.ndim == 2:
            for j in range(2):
                for i in range(2):
                    He = self.He[:, :, 2 * j + i]
                    detJ = self.detJ[:, 2 * j + i]

                    eu = He @ ue
                    ev = He @ ve
                    dfdrhoE += detJ * np.einsum("nik,nik -> n", eu, ev)

        if self.ptype_M == "msimp":
            dnonlin = 6.0 * self.simp_c1 * rhoE**5.0 + 7.0 * self.simp_c2 * rhoE**6.0
            cond = (rhoE > 0.1).astype(int)
            dfdrhoE[:] *= self.density * (cond + dnonlin * (1 - cond))
        elif self.ptype_M == "ramp":
            dfdrhoE[:] *= self.density * (1.0 + self.q) / (1.0 + self.q * rhoE) ** 2
        else:  # linear
            dfdrhoE[:] *= self.density

        dMdrho = np.zeros(self.nnodes)

        for i in range(4):
            np.add.at(dMdrho, self.conn[:, i], dfdrhoE)
        dMdrho *= 0.25

        return dMdrho

    def get_stress_stiffness_matrix(self, rhoE, u):
        """
        Assemble the stess stiffness matrix
        """

        # Get the element-wise solution variables
        ue = np.zeros((self.nelems, 8), dtype=rhoE.dtype)
        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        # Compute the element stiffnesses
        if self.ptype_G == "simp":
            C = np.outer(rhoE**self.p + self.rho0_G, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_G, self.C0)

        C = C.reshape((self.nelems, 3, 3))

        Ge = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)

        for i in range(4):
            detJ = self.detJ[:, i]
            Be = self.Be[:, :, :, i]
            Te = self.Te[:, :, :, :, i]

            # Compute the stresses in each element
            s = np.einsum("nij,njk,nk -> ni", C, Be, ue)

            G0e = np.einsum("n,ni,nijl -> njl", detJ, s, Te)
            Ge[:, 0::2, 0::2] += G0e
            Ge[:, 1::2, 1::2] += G0e

        G = sparse.coo_matrix((Ge.flatten(), (self.i, self.j)))
        G = G.tocsr()

        return G

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
        u=None,
        tol=1e-12,
        maxiter=100,
        k_max=10,
        lmax=None,
        geteigval=False,
    ):
        if u is None:
            u = np.zeros(2 * self.nnodes)

        # store the u and l with size (2*nnodes, n_max)
        u_list = []
        l_list = []
        eig_Kt_list = []

        self.setBCs(u)
        u_prev = u.copy()
        u_prev_prev = u.copy()

        l, l_prev, l_prev_prev = Dl, 0.0, 0.0
        Ds, Ds_prev, Ds_max, Ds_min = Dl, Dl, Dl, Dl / 1024

        converged = False
        converged_prev = False

        ff = np.dot(self.f, self.f)
        fr = self.reduce_vector(self.f).reshape(-1, 1)

        # arch-length algorithm
        for n in range(maxiter):
            if n > 0:
                a = Ds / Ds_prev
                u = (1 + a) * u_prev - a * u_prev_prev
                l = (1 + a) * l_prev - a * l_prev_prev

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
                    Ds_min, Ds_max = Ds / 1024, Ds

                l_prev_prev = l_prev
                l_prev = l
                u_prev_prev = u_prev
                u_prev = u
                Ds_prev = Ds

                if converged_prev:
                    Ds = min(max(2.0 * Ds, Ds_min), Ds_max)

            else:
                if converged_prev:
                    Ds = max(Ds / 2, Ds_min)
                else:
                    Ds = max(Ds / 4, Ds_min)

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

        if geteigval:
            return u_list, l_list, eig_Kt_list
        else:
            return u_list, l_list

    def approximate_critical_load_factor(
        self, sigma, lim=None, tol=1e-8, max_iter=100, eig_method=0
    ):
        """
        Approximate the critical load factor using the Newton-Raphson method.
        """
        # compute u1 at lam = 1.0
        u1 = self.newton_raphson(1.0, tol=tol, max_iter=max_iter)

        # compute K and G for u=u1
        K_li, G_nl = self.getKt(self.rhoE, u1, return_G=True)

        # Apply boundary conditions for K and G
        Kr = self.reduce_matrix(K_li)
        Gr = self.reduce_matrix(G_nl)

        # compute the eigenvalues and eigenvectors
        if eig_method == 1:
            mu, u1 = sparse.linalg.eigsh(Gr, M=Kr, k=1, which="SM", igma=sigma)
            lam_c = -1.0 / mu[0][0]
        else:
            factor = SpLuOperator(Kr + sigma * Gr)
            eig_solver = IRAM(N=1, m=2, mode="buckling", tol=tol)
            lam_c, u1 = eig_solver.solve(Gr, Kr, factor, sigma)

        # bound the critical load factor by the limit
        if lim is not None:
            lam_c = np.clip(lam_c, 0.1 * lim, lim)
        else:
            lam_c = np.clip(lam_c, 0.1 * sigma, 1.1 * sigma)

        u1 = self.full_vector(u1[:, 0])
        return lam_c[0], u1

    def approximate_u1(self, lam_c):
        print(f"Approximate u1 at lam = {lam_c}")
        u = self.newton_raphson(lam_c)
        # u = u / np.linalg.norm(u)
        return u

    def get_koiter_ab(self, rhoE, lam_c, u0, u1):
        # normlize u1
        u1 = u1 / np.linalg.norm(u1)

        # Get the element-wise solution variables
        ue0 = np.zeros((self.nelems, 8), dtype=rhoE.dtype)
        ue0[:, ::2] = u0[2 * self.conn]
        ue0[:, 1::2] = u0[2 * self.conn + 1]

        ue1 = np.zeros((self.nelems, 8), dtype=rhoE.dtype)
        ue1[:, ::2, ...] = u1[2 * self.conn, ...]
        ue1[:, 1::2, ...] = u1[2 * self.conn + 1, ...]

        # Compute the element stiffnesses
        if self.ptype_K == "simp":
            CK = np.outer(rhoE**self.p + self.rho0_K, self.C0)
            CG = np.outer(rhoE**self.p + self.rho0_G, self.C0)
        else:  # ramp
            CK = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_K, self.C0)
            CG = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_G, self.C0)

        CK = CK.reshape((self.nelems, 3, 3))
        CG = CG.reshape((self.nelems, 3, 3))

        Ke0 = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)
        Ke1 = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)
        Ke11 = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)

        Ge0 = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)
        Ge1 = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)

        t0 = time.time()
        for i in range(4):
            detJ = self.detJ[:, i]
            Be = self.Be[:, :, :, i]
            Te = self.Te[:, :, :, :, i]

            # strain_nl = np.zeros((self.nelems, 3))
            Be1 = np.zeros((self.nelems, 3, 8))
            populate_nonlinear_strain_and_Be(Be, ue1, Be1)

            CKBe = CK @ Be
            CKBe1 = CK @ Be1

            Ke0 += detJ.reshape(-1, 1, 1) * Be.transpose(0, 2, 1) @ CKBe
            Ke1 += detJ.reshape(-1, 1, 1) * Be.transpose(0, 2, 1) @ CKBe1
            Ke11 += detJ.reshape(-1, 1, 1) * Be1.transpose(0, 2, 1) @ CKBe1

            # populate_nonlinear_strain_and_Be(Be, ue0, Be1, strain_nl)
            s = np.einsum("nik,nk -> ni", CKBe, ue0)
            # s = np.einsum("nij,njk,nk -> ni", CK, Be, ue0)
            G0e = np.einsum("n,ni,nijl -> njl", detJ, s, Te)
            Ge0[:, 0::2, 0::2] += G0e
            Ge0[:, 1::2, 1::2] += G0e

            # populate_nonlinear_strain_and_Be(Be, ue1, Be1, strain_nl)
            # s1 = np.einsum("nij,nj -> ni", CG, strain_nl)
            s1 = np.einsum("nik,nk -> ni", CKBe, ue1)
            # s1 = np.einsum("nij,njk,nk -> ni", CK, Be, ue1)
            G0e1 = np.einsum("n,ni,nijl -> njl", detJ, s1, Te)
            Ge1[:, 0::2, 0::2] += G0e1
            Ge1[:, 1::2, 1::2] += G0e1

        t1 = time.time()
        ic(t1 - t0)

        # Create sparse matrices directly
        K0 = sparse.csc_matrix((Ke0.flatten(), (self.i, self.j)))
        K1 = sparse.csc_matrix((Ke1.flatten(), (self.i, self.j)))
        K11 = sparse.csc_matrix((Ke11.flatten(), (self.i, self.j)))
        G0 = sparse.csc_matrix((Ge0.flatten(), (self.i, self.j)))
        G1 = sparse.csc_matrix((Ge1.flatten(), (self.i, self.j)))

        K0r = self.reduce_matrix(K0)
        K1r = self.reduce_matrix(K1)
        G0r = self.reduce_matrix(G0)
        G1r = self.reduce_matrix(G1)
        u1r = self.reduce_vector(u1)

        # Formulate block matrix directly using sparse operations
        Ar = K0r + lam_c * G0r
        L = (K0r @ u1r).reshape(-1, 1)  # SAME: L = (K1r @ u0r).reshape(-1, 1)
        a = sparse.csc_matrix(([0], ([0], [0])), shape=(1, 1))
        rhs = -(G1r + 0.5 * K1r) @ u1r

        # Construct the block matrix A in sparse format
        A = sparse.bmat([[Ar, L], [L.T, a]], format="csc")

        # Solve the linear system
        x = sparse.linalg.spsolve(A, np.hstack([rhs, 0]))
        u2 = self.full_vector(x[:-1])

        ue2 = np.zeros((self.nelems, 8), dtype=rhoE.dtype)
        ue2[:, ::2, ...] = u2[2 * self.conn, ...]
        ue2[:, 1::2, ...] = u2[2 * self.conn + 1, ...]

        # check if u1 orthogonal to u2
        a22 = 0.0
        b22 = 0.0
        a11 = 0.0
        b11 = 0.0

        an = 0.0
        bn = 0.0
        d1 = 0.0
        d2 = 0.0

        t0 = time.time()
        for i in range(4):
            detJ = self.detJ[:, i]
            Be = self.Be[:, :, :, i]
            Be1 = np.zeros((self.nelems, 3, 8))
            populate_nonlinear_strain_and_Be(Be, ue1, Be1)

            # strain_0 = np.einsum("nij,nj -> ni", Be, ue0)
            strain_1 = np.einsum("nij,nj -> ni", Be, ue1)
            strain_11 = np.einsum("nij,nj -> ni", Be1, ue1)
            strain_12 = np.einsum("nij,nj -> ni", Be1, ue2)
            strain_02 = np.einsum("nij,nj -> ni", Be, ue2)

            # stress_0 = np.einsum("nij,nj -> ni", CK, strain_0)
            stress_1 = np.einsum("nij,nj -> ni", CK, strain_1)
            stress_2 = np.einsum("nij,nj -> ni", CK, (strain_02 + 0.5 * strain_11))

            an += np.einsum("ni,ni -> n", stress_1, strain_11)
            b1 = np.einsum("ni,ni -> n", stress_2, strain_11)
            b2 = np.einsum("ni,ni -> n", stress_1, strain_12)
            bn += b1 + 2 * b2
            d2 += np.einsum("ni,ni -> n", stress_1, strain_1)

        an = np.sum(an)
        bn = np.sum(bn)
        # d1 = np.sum(d1)
        d2 = np.sum(d2)

        # a11 = 1.5 * an / d1
        a22 = 1.5 * an / d2

        # b11 = bn / d1
        b22 = bn / d2
        t1 = time.time()
        ic(t1 - t0)

        ic(a11, a22)
        ic(b11, b22)

        a = a11
        b = b11
        t0 = time.time()
        u2r = self.reduce_vector(u2)
        K11r = self.reduce_matrix(K11)

        a1 = u1r.T @ K0r @ u1r
        a2 = K1r @ u1r
        a = 1.5 * (u1r.T @ a2) / a1
        b = (u2r.T @ a2 + 2 * u1r.T @ K1r @ u2r + 0.5 * u1r.T @ K11r @ u1r) / a1
        t1 = time.time()
        ic(t1 - t0)

        ic(a, b)
        # exit()

        indy = 2 * np.nonzero(self.f[1::2])[0] + 1
        indy = indy[len(indy) // 2]
        indx = indy - 1

        xi = np.linspace(-1e1, 1e1, 100)

        lam = (1 + a * xi + b * xi**2) * lam_c

        ux = u0[indx] * lam + u1[indx] * xi + u2[indx] * xi**2
        uy = u0[indy] * lam + u1[indy] * xi + u2[indy] * xi**2

        lam_0 = np.linspace(0, lam_c, 100)
        u0x = u0[indx] * lam_0
        u0y = u0[indy] * lam_0

        fig, ax = plt.subplots(1, 4, figsize=(16, 3), tight_layout=True)
        ax[0].plot(xi, lam / lam_c, color="k")
        ax[0].plot([0, 0], [0, 1], color="b")
        ax[0].scatter(0, 1, color="r", zorder=10)
        ax[0].set_xlabel(r"$\xi$")
        ax[0].set_ylabel(r"$\lambda/\lambda_c$")
        ax[0].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        ax[1].plot(u0x, lam_0, color="b")
        ax[1].plot(ux, lam, color="k")
        ax[1].scatter(u0[indx] * lam_c, lam_c, color="r", zorder=10)
        ax[1].set_xlabel(r"$u_x$")
        ax[1].set_ylabel(r"$\lambda$")
        ax[1].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        ax[2].plot(u0y, lam_0, color="b")
        ax[2].plot(uy, lam, color="k")
        ax[2].scatter(u0[indy] * lam_c, lam_c, color="r", zorder=10)
        ax[2].set_xlabel(r"$u_y$")
        ax[2].set_ylabel(r"$\lambda$")
        ax[2].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax[2].invert_xaxis()

        res_norm0 = np.zeros(len(xi))
        res_norm = np.zeros(len(xi))
        # for i in range(len(xi)):
        #     ui = u0 * lam[i] + u1 * xi[i] + u2 * xi[i] ** 2
        #     u0i = u0 * lam[i]
        #     res = self.getResidual(self.rhoE, ui, lam[i])[1]
        #     res_norm[i] = np.linalg.norm(res)

        #     res = self.getResidual(self.rhoE, u0i, lam_c)[1]
        #     res_norm0[i] = np.linalg.norm(res)
        #     print(f"Residual[{i:3d}]  {res_norm[i]:15.5e}")

        ax[3].semilogy(xi, res_norm, color="b")
        ax[3].semilogy(xi, res_norm0, color="k")
        ax[3].set_xlabel(r"$\xi$")
        ax[3].set_ylabel(r"$||R||$")
        ax[3].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        plt.savefig("load-deflection.pdf", bbox_inches="tight")

        def plot_u():
            xi = 1e1
            lam = (1 + a * xi + b * xi**2) * lam_c
            u = u0 * lam + u1 * xi + u2 * xi**2
            u_lin = u0 * lam
            u_list = [u0, u1, u0 * lam_c, u_lin, u]

            fig, ax = plt.subplots(1, 5, figsize=(15, 4))
            levels = np.linspace(0.0, 1.0, 26)
            title = [
                r"$u_0$",
                r"$u_1$",
                r"$u_0 \lambda_c$",
                r"$u_0 \lambda$",
                r"$u$ at $\lambda = " + f"{lam:.2f}$",
            ]
            for i, u in enumerate(u_list):
                self.plot(
                    self.rho, ax=ax[i], u=u, levels=levels, extend="max", cmap="Greys"
                )
                ax[i].set_title(title[i])
                ax[i].set_xticks([])
                ax[i].set_yticks([])

            plt.savefig("u.pdf", bbox_inches="tight")

        plot_u()

        self.K0 = K0
        self.K1 = K1
        self.G0 = G0
        self.G1 = G1
        self.K11 = K11

        return a, b, u1, u2

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
        K = self.get_stiffness_matrix(self.rhoE)
        G = self.get_stress_stiffness_matrix(self.rhoE, u0)
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

    def get_tangent_stiffness_matrix(self, rhoE, u):

        # Get the element-wise solution variables
        ue = np.zeros((self.nelems, 8), dtype=rhoE.dtype)
        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        # Compute the element stiffnesses
        if self.ptype_G == "simp":
            C = np.outer(rhoE**self.p + self.rho0_G, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_G, self.C0)

        C = C.reshape((self.nelems, 3, 3))

        # Assemble all of the the 8 x 8 element stiffness matrix
        Ke = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)
        Ke1 = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)
        Ke2 = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)

        Ge = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)
        Ge1 = np.zeros((self.nelems, 8, 8), dtype=rhoE.dtype)

        for i in range(4):
            detJ = self.detJ[:, i]
            Be = self.Be[:, :, :, i]
            Te = self.Te[:, :, :, :, i]

            strain = np.zeros((self.nelems, 3))
            Be_nl = np.zeros((self.nelems, 3, 8))

            populate_nonlinear_strain_and_Be(Be, ue, strain, Be_nl)

            Ke += detJ[:, np.newaxis, np.newaxis] * Be.transpose(0, 2, 1) @ C @ Be
            Ke1 += detJ[:, np.newaxis, np.newaxis] * Be.transpose(0, 2, 1) @ C @ Be_nl
            Ke2 += (
                detJ[:, np.newaxis, np.newaxis] * Be_nl.transpose(0, 2, 1) @ C @ Be_nl
            )

            s = np.eisnum("nij,nj -> ni", C, strain)
            G0e = np.einsum("n,ni,nijl -> njl", detJ, s, Te)
            Ge1[:, 0::2, 0::2] += G0e
            Ge1[:, 1::2, 1::2] += G0e

    def intital_Be_and_Te(self):
        # Compute gauss points
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        # Compute Be and Te, detJ
        Be = np.zeros((self.nelems, 3, 8, 4))
        Te = np.zeros((self.nelems, 3, 4, 4, 4))
        detJ = np.zeros((self.nelems, 4))

        for j in range(2):
            for i in range(2):
                xi, eta = gauss_pts[i], gauss_pts[j]
                index = 2 * j + i
                Bei = Be[:, :, :, index]
                Tei = Te[:, :, :, :, index]

                detJ[:, index] = populate_Be_and_Te(
                    self.nelems, xi, eta, xe, ye, Bei, Tei
                )

        return Be, Te, detJ

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

        return dfdrho

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

        return dfdrho

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

    def solve_eigenvalue_problem(self, rhoE, store=False):
        """
        Compute the smallest buckling load factor BLF
        """

        t0 = time.time()

        K = self.get_stiffness_matrix(rhoE)
        self.Kr = self.reduce_matrix(K)

        # Compute the solution path
        fr = self.reduce_vector(self.f)
        self.Kfact = linalg.factorized(self.Kr.tocsc())
        ur = self.Kfact(fr)
        self.u = self.full_vector(ur)

        # Find the gemoetric stiffness matrix
        G = self.get_stress_stiffness_matrix(rhoE, self.u)
        self.Gr = self.reduce_matrix(G)

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
        return self.fltr.apply_gradient(dfdrho, self.x)

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

        dfdrho = np.zeros(self.nnodes)
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

                # dGdx = self.get_stress_stiffness_matrix_deriv(
                #     self.rhoE, self.u, self.Q[:, i], self.Q[:, i], self.Kfact
                # )

                dfdrho -= eta[i] * (dGdx + mu[i] * dKdx)

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

            dfdrho -= dGdx + dKdx

        t1 = time.time()
        self.profile["total derivative time"] += t1 - t0

        return self.fltr.apply_gradient(dfdrho, self.x)

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

        # Apply the filter
        # self.rho = self.fltr.apply(self.x)

        # Average the density to get the element-wise density
        self.rhoE = 0.25 * (
            self.rho[self.conn[:, 0]]
            + self.rho[self.conn[:, 1]]
            + self.rho[self.conn[:, 2]]
            + self.rho[self.conn[:, 3]]
        )

        self.Be, self.Te, self.detJ = self.intital_Be_and_Te()

        # Solve the eigenvalue problem
        self.lam, self.Q = self.solve_eigenvalue_problem(self.rhoE, store)

        # self.lam = self.lam[1:]
        # self.Q = self.Q[:, 1:]

        print("Eigenvalues: ", self.lam)
        if store:
            self.profile["eigenvalues"] = self.BLF.tolist()

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

        self.xb += self.fltr.apply_gradient(self.rhob, self.x)

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

        rho = self.fltr.apply(self.x)

        # Average the density to get the element-wise density
        rhoE = 0.25 * (
            rho[self.conn[:, 0]]
            + rho[self.conn[:, 1]]
            + rho[self.conn[:, 2]]
            + rho[self.conn[:, 3]]
        )

        K = self.get_stiffness_matrix(rhoE)
        Kr = self.reduce_matrix(K)

        # Compute the solution path
        fr = self.reduce_vector(self.f)
        Kfact = linalg.factorized(Kr.tocsc())
        ur = Kfact(fr)
        u = self.full_vector(ur)

        # Find the gemoetric stiffness matrix
        G = self.get_stress_stiffness_matrix(rhoE, u)

        kx = self.get_stiffness_matrix_deriv(rhoE, psi, phi)
        gx = self.get_stress_stiffness_matrix_deriv(rhoE, u, psi, phi, Kfact)

        g_f0 = np.dot(psi, G @ phi)
        k_f0 = np.dot(psi, K @ phi)

        p_rho = np.random.uniform(size=rho.shape)
        rho_1 = rho + dh * p_rho

        g_exact = np.dot(gx, p_rho)
        k_exact = np.dot(kx, p_rho)

        rhoE_1 = 0.25 * (
            rho_1[self.conn[:, 0]]
            + rho_1[self.conn[:, 1]]
            + rho_1[self.conn[:, 2]]
            + rho_1[self.conn[:, 3]]
        )

        K = self.get_stiffness_matrix(rhoE_1)
        Kr = self.reduce_matrix(K)

        # Compute the solution path
        fr = self.reduce_vector(self.f)
        Kfact = linalg.factorized(Kr.tocsc())
        ur = Kfact(fr)
        u = self.full_vector(ur)

        # Find the gemoetric stiffness matrix
        G = self.get_stress_stiffness_matrix(rhoE_1, u)
        K = self.get_stiffness_matrix(rhoE_1)

        k_f1 = np.dot(psi, K @ phi)
        k_fd = (k_f1 - k_f0) / dh

        print("\nK: Exact: ", k_exact)
        print("K: FD: ", k_fd)
        print("K: Error: ", np.abs(k_exact - k_fd) / np.abs(k_exact))

        g_f1 = np.dot(psi, G @ phi)
        g_fd = (g_f1 - g_f0) / dh

        print("\nG: Exact: ", g_exact)
        print("G: FD: ", g_fd)
        print("G: Error: ", np.abs(g_exact - g_fd) / np.abs(g_exact))

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

    def test_ks_buckling_derivatives(self, dh_fd=1e-4, ks_rho=30, pert=None):
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

        # plot the nodes
        for i in range(self.nnodes):
            ax.scatter(self.X[i, 0], self.X[i, 1], color="k", s=1, clip_on=False)

        # plot the conn
        for i in range(self.nelems):
            for j in range(4):
                x0 = self.X[self.conn[i, j], 0]
                y0 = self.X[self.conn[i, j], 1]
                x1 = self.X[self.conn[i, (j + 1) % 4], 0]
                y1 = self.X[self.conn[i, (j + 1) % 4], 1]
                ax.plot([x0, x1], [y0, y1], color="k", linewidth=0.5, clip_on=False)

        # plot the bcs
        for i, v in self.bcs.items():
            ax.scatter(self.X[i, 0], self.X[i, 1], color="b", s=5, clip_on=False)

        for i, v in self.forces.items():
            ax.quiver(
                self.X[i, 0],
                self.X[i, 1],
                v[0],
                v[1],
                color="r",
                scale=1e-3,
                clip_on=False,
            )

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

    def plot_u_compare(self, u1, l1, u2, l2, path=None):
        # get the middle node of the index where the force is applied
        indy = 2 * np.nonzero(self.f[1::2])[0] + 1
        indy = indy[len(indy) // 2]
        indx = indy - 1

        fig, ax = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
        ax[0].plot(u1[:, indx], l1 / self.lam[0], marker="o", label="Arc-Length")
        ax[0].plot(u2[:, indx], l2 / self.lam[0], marker="o", label="Koiter-Asymptotic")
        ax[0].set_xlabel(r"$u_x$")
        ax[0].set_ylabel(r"$\lambda/\lambda_c$")
        ax[0].axvline(x=0, color="grey", linestyle="--")
        ax[0].axhline(y=1, color="grey", linestyle="--")
        ax[0].legend()

        ax[1].plot(u1[:, indy], l1 / self.lam[0], marker="o", label="Arc-Length")
        ax[1].plot(u2[:, indy], l2 / self.lam[0], marker="o", label="Koiter-Asymptotic")
        ax[1].set_xlabel(r"$u_y$")
        ax[1].set_ylabel(r"$\lambda/\lambda_c$")
        ax[1].axvline(x=0, color="k", linestyle="--")
        ax[1].axhline(y=1, color="k", linestyle="--")
        ax[1].invert_xaxis()
        ax[1].legend()

        plt.savefig(path, bbox_inches="tight")

        return ax


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

    non_design_nodes = index
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

    return conn, X, dvmap, non_design_nodes, bcs, forces


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

    non_design_nodes = []
    # for jp in range(nt - nm, nt + 1):
    #     for ip in range(nx - nm, nx + 1):
    #         non_design_nodes.append(ij_to_node(ip, jp))

    bcs = {}
    # # top left corner ---- fixed
    # for ip in range(nt + 1):
    #     bcs[ij_to_node(ip, nx)] = [0, 1]
    # non_design_nodes.append(ij_to_node(ip, nx))
    # non_design_nodes.append(ij_to_node(ip, nx-1))

    # bottom left corner ____ fixed
    for ip in range(nt + 1):
        bcs[ij_to_node(ip, 0)] = [0, 1]

    # top right corner | fixed
    for jp in range(nx - nt, nx + 1):
        bcs[ij_to_node(nx, jp)] = [0, 1]

    forces = {}
    P = -5e-4
    P_nnodes = int(np.ceil(0.01 * nx))
    P_pst = int(np.ceil(0.2 * nx))
    P_pst = 0
    P_nnodes = nt + 1
    for ip in range(P_pst, P_pst + P_nnodes):
        forces[ij_to_node(ip, nx)] = [0, P / P_nnodes]

    dvmap = None

    return conn, X, dvmap, non_design_nodes, bcs, forces


def make_model(
    nx=64, ny=128, Lx=1.0, Ly=2.0, rfact=4.0, N=10, shear_force=False, **kwargs
):
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
    ny = int((Ly / Lx) * nx)

    if kwargs.get("domain") == "column":
        conn, X, dvmap, non_design_nodes, bcs, forces = domain_compressed_column(
            nx=nx, ny=ny, Lx=Lx, Ly=Ly, shear_force=shear_force
        )
    elif kwargs.get("domain") == "rooda":
        conn, X, dvmap, non_design_nodes, bcs, forces = domain_rooda_frame(
            nx=nx, l=Lx, lfrac=0.03
        )

    fltr = NodeFilter(
        conn,
        X,
        r0=rfact * (Lx / nx),
        dvmap=dvmap,
        num_design_vars=non_design_nodes,
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
            "update_guess": False,
            "bs_target": 1,
        }

    solver_type = "BasicLanczos"
    if "IRAM" in sys.argv:
        solver_type = "IRAM"

    print("method = ", method)
    print("adjoint_options = ", adjoint_options)
    print("solver_type = ", solver_type)

    topo = make_model(
        nx=60,  # 32 x 64 mesh
        Lx=8.0,
        rfact=4.0,
        N=10,
        sigma=3.0,
        solver_type=solver_type,
        adjoint_method=method,
        adjoint_options=adjoint_options,
        shear_force=False,  # True,
        deriv_type="vector",
        domain="rooda",
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
    topo.testR()
    topo.testKt()

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

    u_list, lam_list, eigvals = topo.arc_length_method(
        Dl=topo.lam[0] * 0.5, lmax=topo.lam[0] * 5, geteigval=True, maxiter=100
    )
    np.save("./output/u_list.npy", u_list)
    np.save("./output/lam_list.npy", lam_list)
    np.save("./output/eigvals.npy", eigvals)

    u_newton = topo.newton_raphson(lam=max(lam_list))
    np.save("./output/u_newton.npy", u_newton)

    u_list = np.load("./output/u_list.npy")
    lam_list = np.load("./output/lam_list.npy")
    eigvals = np.load("./output/eigvals.npy")
    u_newton = np.load("./output/u_newton.npy")

    # plot the u as a video that shows the buckling process
    topo.video_u(u_list, lam_list, "arc_length.mp4", "Arc-Length")
    topo.plot_u(u_list, lam_list, eigvals, "arc_length.pdf")

    # u1, _= topo.arc_length_method(Dl=topo.lam[0] * 0.1, lmax=lam_c)

    # a, b, u1, u2 = topo.get_koiter_ab(topo.rhoE, lam_c, u1, topo.Q[:, 0])
    # u0 = topo.u / np.linalg.norm(topo.u)
    # lam_c = 1.71
    # u1 = topo.newton_raphson(lam=lam_c)

    lam_c = topo.lam[0]
    u0 = topo.u
    u1 = topo.Q[:, 0]
    print(u1.shape)

    # lam_c, u1 = topo.approximate_critical_load_factor(sigma=topo.lam[0])

    # u1 = topo.approximate_u1(lam_c)
    # print("lam_c = ", lam_c)
    # find the index where lam start to decrease for lam_list
    # index_c = 0
    # for i in range(1, lam_list.size):
    #     if lam_list[i] < lam_list[i - 1]:
    #         index_c = i
    #         break
    # lam_c = lam_list[index_c]
    # u1 = u_list[index_c, :]

    a, b, u1, u2 = topo.get_koiter_ab(topo.rhoE, lam_c, u0, u1)
    topo.check_koiter_ab(lam_c, a, b, u0, u1, u2)

    print("a = ", a)
    print("b = ", b)

    xi = np.linspace(0, -1.2e1, 40)
    lam = (1 + a * xi + b * xi**2) * lam_c
    u = np.zeros((lam.size, topo.nvars))
    for i in range(lam.size):
        u[i, :] = lam[i] * u0 + xi[i] * u1 + xi[i] ** 2 * u2

    # from 0 to lam_c is linear
    lam_0 = np.linspace(0, lam_c, 20)
    u_0 = np.zeros((lam_0.size, topo.nvars))
    for i in range(1, lam_0.size):
        u_0[i, :] = lam_0[i] * u0

    # add u_0 to u and lam_0 to lam
    u = np.vstack((u_0, u))
    lam = np.hstack((lam_0, lam))

    topo.video_u(u, lam, "koiter.mp4", "Koiter-Asymptotic")
    topo.plot_u_compare(u_list, lam_list, u, lam, "comparsion.pdf")

    fig, ax = plt.subplots(1, 5, figsize=(15, 5))
    levels = np.linspace(0.0, 1.0, 26)
    uu = [
        u_list[lam_list.size // 4 * 1, :],
        u_list[lam_list.size // 4 * 2, :],
        u_list[lam_list.size // 4 * 3, :],
        u_list[-1, :],
        u_newton,
    ]
    for i in range(5):
        ii = lam_list.size // 4 * (i + 1)
        topo.plot(
            topo.rho,
            u=uu[i],
            ax=ax[i],
            levels=levels,
            cmap="Greys",
            extend="max",
        )
        if i < 3:
            ax[i].title.set_text(r"Arc-Length $\lambda$" + f"= %.2f" % lam_list[ii])
        elif i == 3:
            ax[i].title.set_text(r"Arc-Length $\lambda$" + f"= %.2f" % lam_list[-1])
        else:
            ax[i].title.set_text(r"Newton-Raphson $\lambda$" + f"= %.2f" % lam_list[-1])

        ax[i].set_aspect("equal")
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.savefig("u_history.pdf", bbox_inches="tight")

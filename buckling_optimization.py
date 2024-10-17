import os
import time

import numpy as np


class BucklingOpt:

    def __init__(self, topo, ks_rho=160, rho_agg=100, node=[]):

        self.topo = topo

        self.ks_rho = ks_rho
        self.rho_agg = rho_agg
        self.node = node

        self.hb = 1.0
        self.mode = "tanh"

        if len(self.node) == 0:
            self.node = np.arange(self.topo.nnodes)

        return

    def initialize(self):
        self.topo.initialize()
        self.BLF = self.topo.BLF
        self.l0 = self.topo.lam[0]
        self.Q0 = self.topo.Q[:, 0]
        return

    def initialize_koiter(self, Q0_norm=None):
        self.Q0_norm = Q0_norm
        self.topo.initialize_koiter(self.Q0_norm)
        self.a = self.topo.a
        self.b = self.topo.b

        self.dl = None
        self.da = None
        self.db = None
        return

    def initialize_adjoint(self):
        self.topo.initialize_adjoint()
        return

    def finalize_adjoint(self):
        self.topo.finalize_adjoint()
        return

    def get_ks_buckling(self):
        return self.topo.eval_ks_buckling(self.ks_rho)

    def get_ks_buckling_derivative(self):
        return self.topo.eval_ks_buckling_derivative(self.ks_rho)

    def get_compliance(self):
        return self.topo.compliance()

    def get_compliance_derivative(self):
        return self.topo.compliance_derivative()

    def get_compliance2(self):
        return self.topo.get_compliance()

    def get_compliance_derivative2(self):
        return self.topo.add_compliance_derivative()

    def get_eigenvector_aggregate(self):
        return self.topo.get_eigenvector_aggregate(self.rho_agg, self.node, self.mode)

    def get_eigenvector_aggregate_derivative(self):
        return self.topo.add_eigenvector_aggregate_derivative(
            self.hb, self.rho_agg, self.node, self.mode
        )

    def get_eigenvector_aggregate_max(self):
        return self.topo.get_eigenvector_aggregate_max(self.rho_agg, self.node)

    def get_eigenvector_aggregate_max_derivative(self):
        return self.topo.add_eigenvector_aggregate_max_derivative(
            self.hb, self.rho_agg, self.node
        )

    def get_area(self):
        return self.topo.eval_area()

    def get_area_derivative(self):
        return self.topo.eval_area_gradient()

    def get_koiter_a(self):
        return np.abs(self.a)

    def get_dadx(self):
        t1 = time.time()

        if self.da is None:
            self.da = self.topo.get_dadx(self.topo.rhoE, self.topo.l0, self.topo.Q0)

        t2 = time.time()
        self.topo.profile["koiter time"] += t2 - t1

        return self.da

    def get_dbdx(self):
        t1 = time.time()

        if self.db is None:
            self.db = self.topo.get_dbdx(self.topo.rhoE, self.topo.l0, self.topo.Q0)

        t2 = time.time()
        self.topo.profile["koiter time"] += t2 - t1

        return self.db

    def get_dldx(self):
        if self.dl is None:
            self.dl = self.topo.get_dldx()
        return self.dl

    def get_koiter_da(self):
        da = self.get_dadx()
        if self.a < 0:
            da = -da
        return da

    def get_koiter_b(self):
        return self.b

    def get_koiter_db(self):
        return self.get_dbdx()

    def get_koiter_al0(self):

        al0 = self.l0 / self.a

        if self.a < 0:
            al0 *= -1

        return al0

    def get_koiter_dal0(self):
        da = self.get_dadx()
        dl0 = self.get_dldx()

        dal0dx = dl0 / self.a - self.l0 / self.a**2 * da

        if self.a < 0:
            return -dal0dx

        return dal0dx

    def get_koiter_lams(self, xi=1e-3):
        if self.Q0_norm is None:
            Q0_norm = np.linalg.norm(self.Q0)
        xi = xi * Q0_norm
        return self.topo.get_lam_s(self.l0, self.a, xi)

    def get_koiter_dlams(self, xi=1e-3):
        da = self.get_dadx()
        dl = self.get_dldx()

        if self.Q0_norm is None:
            Q0_norm = np.linalg.norm(self.Q0)
            xi = xi * Q0_norm
        return self.topo.get_dlamsdx(self.l0, self.a, dl, da, xi)

    def get_koiter_ks_lams(self, xi=1e-3):
        if self.Q0_norm is None:
            Q0_norm = np.linalg.norm(self.Q0)
            xi = xi * Q0_norm
            print("xi:", xi)
        return self.topo.get_ks_lams(self.a, xi, self.ks_rho)

    def get_koiter_ks_dlams(self, xi=1e-3):
        if self.Q0_norm is None:
            Q0_norm = np.linalg.norm(self.Q0)
            xi = xi * Q0_norm
        da = self.get_dadx()
        return self.topo.get_ks_lams_derivatives(self.a, da, xi, self.ks_rho)

    def get_koiter_normalized_lams(self, xi=1e-3):
        if self.Q0_norm is None:
            Q0_norm = np.linalg.norm(self.Q0)
            xi = xi * Q0_norm
        return self.topo.get_flam_s(self.l0, self.a, xi)

    def get_koiter_normalized_dlams(self, xi=1e-3):
        da = self.get_dadx()
        if self.Q0_norm is None:
            Q0_norm = np.linalg.norm(self.Q0)
            xi = xi * Q0_norm
        return self.topo.get_dflamsdx(self.a, da, xi)

    def get_koiter_lams_b(self, xi=1e-3):
        if self.Q0_norm is None:
            Q0_norm = np.linalg.norm(self.Q0)
            xi = xi * Q0_norm
        return self.topo.get_lams_b(self.l0, self.b, xi)

    def get_koiter_dlams_b(self, xi=1e-3):
        dl = self.get_dldx()
        db = self.get_dbdx()
        if self.Q0_norm is None:
            Q0_norm = np.linalg.norm(self.Q0)
            xi = xi * Q0_norm
        return self.topo.get_dlams_b(self.l0, self.b, dl, db, xi)

    def get_koiter_ks_lams_b(self, xi=1e-3):
        if self.Q0_norm is None:
            Q0_norm = np.linalg.norm(self.Q0)
            xi = xi * Q0_norm
        return self.topo.get_ks_lams_b(self.b, xi, self.ks_rho)

    def get_koiter_ks_dlams_b(self, xi=1e-3):
        if self.Q0_norm is None:
            Q0_norm = np.linalg.norm(self.Q0)
            xi = xi * Q0_norm
        db = self.get_dbdx()
        return self.topo.get_ks_lams_b_derivatives(self.b, db, xi, self.ks_rho)

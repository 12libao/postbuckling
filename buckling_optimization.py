import os

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

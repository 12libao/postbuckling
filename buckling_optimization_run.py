import os

from icecream import ic
import mpi4py.MPI as MPI
import numpy as np
from paropt import ParOpt

from buckling import make_model
from buckling_optimization import BucklingOpt
from utils import Logger, get_args


class ParOptProb(ParOpt.Problem):

    def __init__(self, comm, prob: BucklingOpt, args, grad_check=False):
        self.comm = comm
        self.prob = prob
        self.args = args

        self.prob.topo.x[:] = 1.0
        self.prob.initialize()

        # compute the Q0 norm, which is fixed for gradient check
        if grad_check:
            self.Q0_norm = np.linalg.norm(self.prob.topo.Q[:, 0])
        else:
            self.Q0_norm = None

        self.domain_area = self.prob.get_area()
        self.area_ub = self.args.vol_frac_ub * self.domain_area

        self.logger = Logger(self, grad_check, "buckling")
        self.logger.initialize_output()

        self.ndvs = self.prob.topo.fltr.num_design_vars
        self.ncon = len(args.confs) if isinstance(args.confs, list) else 1

        super().__init__(comm, nvars=self.ndvs, ncon=self.ncon)

        return

    def getVarsAndBounds(self, x, lb, ub):
        x[:] = 1.0
        lb[:] = self.args.lb
        ub[:] = 1.0
        return

    def evalObjCon(self, x):
        fail = False
        self.obj = []
        con = []

        # Update interpolation parameters
        self.logger.set_interpolation_parameters()

        # Set the design variables and initialize the problem
        self.prob.topo.x[:] = x[:]
        self.prob.initialize()
        self.prob.initialize_koiter(self.Q0_norm)

        # Extract the objective function
        if self.args.objf == "ks-buckling":
            self.obj_scale = 10.0
            self.obj = self.prob.get_ks_buckling()

        elif self.args.objf == "compliance":
            self.obj_scale = 1e5
            self.obj = self.prob.get_compliance()

        elif self.args.objf == "compliance-buckling":
            self.obj_scale = 1.0
            self.c_norm = self.prob.get_compliance() / self.args.c0
            self.ks_norm = self.prob.get_ks_buckling() / self.args.ks0
            self.obj = self.args.w * self.c_norm + (1 - self.args.w) * self.ks_norm

        elif self.args.objf == "aggregate-max":
            self.obj_scale = 1.0
            self.obj = self.prob.get_eigenvector_aggregate_max()

        elif self.args.objf == "koiter-a":
            self.obj_scale = 1.0
            self.obj = self.prob.get_koiter_a()

        elif self.args.objf == "koiter-b":
            self.obj_scale = 1.0
            self.obj = self.prob.get_koiter_b()

        elif self.args.objf == "koiter-al0":
            self.obj_scale = 1e-5
            self.obj = -self.prob.get_koiter_al0()

        elif self.args.objf == "koiter-lams":
            self.obj_scale = 1.0
            self.obj = -self.prob.get_koiter_lams(self.args.xi)

        elif self.args.objf == "koiter-ks-lams":
            self.obj_scale = 10.0
            self.obj = self.prob.get_koiter_ks_lams(self.args.xi)

        elif self.args.objf == "koiter-lams-b":
            self.obj_scale = 1.0
            self.obj = -self.prob.get_koiter_lams_b(self.args.xi)

        elif self.args.objf == "koiter-ks-lams-b":
            self.obj_scale = 10.0
            self.obj = self.prob.get_koiter_ks_lams_b(self.args.xi)

        elif self.args.objf == "koiter-ks-lams-bc":
            self.obj_scale = 10.0
            self.c_norm = self.prob.get_compliance() / self.args.c0
            self.ks_norm = self.prob.get_koiter_ks_lams_b(self.args.xi) / self.args.ks0
            self.obj = self.args.w * self.c_norm + (1 - self.args.w) * self.ks_norm

        elif self.args.objf == "koiter-nlams":
            self.obj_scale = 1.0
            self.obj = -self.prob.get_koiter_normalized_lams(self.args.xi)

        self.obj *= self.obj_scale

        # Evaluate the area constraint
        if "volume" in self.args.confs:
            area = self.prob.get_area()
            self.vol_frac = area / self.domain_area
            con.append(1 - area / self.area_ub)

        if "compliance" in self.args.confs:
            self.c = self.prob.get_compliance()
            con.append(1.0 - self.c / self.args.c_ub)

        if "aggregate" in self.args.confs:
            self.h = self.prob.get_eigenvector_aggregate()
            con.append(1.0 - self.h / self.args.h_ub)

        if "ks-buckling" in self.args.confs:
            self.ks = self.prob.get_ks_buckling()
            self.BLF_ks = 1.0 / self.ks
            con.append(self.BLF_ks / self.args.BLF_ks_lb - 1.0)

        if "koiter-a" in self.args.confs:
            a = self.prob.get_koiter_a()
            con.append(1.0 - a / self.args.a_ub)

        if "koiter-b" in self.args.confs:
            b = self.prob.get_koiter_b()
            con.append(b / self.args.b_lb - 1.0)

        return fail, self.obj, con

    def evalObjConGradient(self, x, g, A):

        # Evaluate the gradient of the objective function
        if self.args.objf == "ks-buckling":
            g0 = self.prob.get_ks_buckling_derivative()

        elif self.args.objf == "compliance":
            g0 = self.prob.get_compliance_derivative()

        elif self.args.objf == "compliance-buckling":
            dc = self.prob.get_compliance_derivative()
            dks = self.prob.get_ks_buckling_derivative()
            g0 = self.args.w * (dc / self.args.c0) + (1 - self.args.w) * (
                dks / self.args.ks0
            )

        elif self.args.objf == "aggregate-max":
            self.prob.initialize_adjoint()
            self.prob.get_eigenvector_aggregate_max_derivative()
            self.prob.finalize_adjoint()

        elif self.args.objf == "koiter-a":
            g0 = self.prob.get_koiter_da()

        elif self.args.objf == "koiter-b":
            g0 = self.prob.get_koiter_db()

        elif self.args.objf == "koiter-al0":
            g0 = -self.prob.get_koiter_dal0()

        elif self.args.objf == "koiter-lams":
            g0 = -self.prob.get_koiter_dlams(self.args.xi)

        elif self.args.objf == "koiter-ks-lams":
            g0 = self.prob.get_koiter_ks_dlams(self.args.xi)

        elif self.args.objf == "koiter-lams-b":
            g0 = -self.prob.get_koiter_dlams_b(self.args.xi)

        elif self.args.objf == "koiter-ks-lams-b":
            g0 = self.prob.get_koiter_ks_dlams_b(self.args.xi)

        elif self.args.objf == "koiter-ks-lams-bc":
            dc = self.prob.get_compliance_derivative()
            dks = self.prob.get_koiter_ks_dlams_b(self.args.xi)
            g0 = self.args.w * (dc / self.args.c0) + (1 - self.args.w) * (
                dks / self.args.ks0
            )

        elif self.args.objf == "koiter-nlams":
            g0 = -self.prob.get_koiter_normalized_dlams(self.args.xi)

        g[:] = g0[:] * self.obj_scale

        index = 0
        if "volume" in self.args.confs:
            A[index][:] = -self.prob.get_area_derivative() / self.area_ub
            index += 1

        if "compliance" in self.args.confs:
            A[index][:] = -self.prob.get_compliance_derivative() / self.args.c_ub
            index += 1

        if "aggregate" in self.args.confs:
            self.prob.initialize_adjoint()
            self.prob.get_eigenvector_aggregate_derivative()
            self.prob.finalize_adjoint()
            A[index][:] = -self.prob.topo.xb / self.args.h_ub
            index += 1

        if "ks-buckling" in self.args.confs:
            dks = self.prob.get_ks_buckling_derivative()
            A[index][:] = -dks / (self.args.BLF_ks_lb * self.ks**2)
            index += 1

        if "koiter-a" in self.args.confs:
            da = self.prob.get_koiter_da()
            A[index][:] = -da / self.args.a_ub
            index += 1

        if "koiter-b" in self.args.confs:
            dbdx = self.prob.get_koiter_db()
            A[index][:] = dbdx / self.args.b_lb
            index += 1

        self.logger.write_output()

        # reset sigma
        self.prob.topo.sigma = self.args.sigma_scale * self.prob.topo.lam[0]

        return False


def settings():
    problem = {
        "domain": "rooda",  # "column" or "rooda"
        "objf": "ks-buckling",
        "w": 0.2,  # weight for compliance
        "c0": 1e-05,  # 1e-5 compliance reference value
        "ks0": 0.06,  # buckling reference value
        "nx": 64,
        "yxratio": 2,
        "ks_rho": 160.0,  # from ferrari2021 paper
        "rho_agg": 100.0,
        "confs": ["volume"],
        "vol_frac_ub": 0.5,
        "BLF_ks_lb": 10.0,
        "a_ub": 1e-4,
        "b_lb": 0.00008,
        "c_ub": 4.3 * 7.4e-6,
        "h_ub": 1.8,
        "lb": 1e-06,  # lower bound of design variables
        "maxiter": 1000,  # maximum number of iterations
    }

    koiter = {
        "xi": -1e-5,
    }

    filer_interpolation = {
        "ptype_K": "simp",  # ramp
        "ptype_M": "simp",  # ramp
        "ptype_G": "simp",  # ramp
        "rho0_K": 1e-6,
        "rho0_M": 1e-9,
        "rho0_G": 1e-6,
        "r": 4.0,
        "p0": 3.0,  # initial value of p
        "q0": 5.0,  # initial value of q
        "psiter": 0,  # start increasing p after this iteration
        "pp": 0.01,  # increase p by this amount every ppiter iterations
        "ppiter": 1,  # increase p every ppiter iterations
        "pmax": 6.0,  # maximum value of p
        "projection": True,  # use projection
        "b0": 1e-6,  # initial value of projection parameter beta
        "bsiter": 0,  # start increasing beta after this iteration
        "bp": 0.1,  # increase beta by this amount every bpiter iterations
        "bpiter": 1,  # increase beta every bpiter iterations
        "bmax": 16.0,  # maximum value of beta
    }

    solver = {
        "N": 6,
        "solver_type": "IRAM",  # IRAM or BasicLanczos
        "adjoint_method": "shift-invert",  # "shift-invert", "ad-adjoint"
        "adjoint_options": {
            "lanczos_guess": True,
            "update_guess": True,
            "bs_target": 1,
        },
        "rtol": 1e-8,
        "eig_atol": 1e-5,
        "sigma": 20.0,
        "sigma_scale": 0.8,
    }

    other = {
        "note": "",
        "prefix": "output",
        "grad_check": False,
        "case_num": 0,  # job submission number for PACE
    }

    settings = [problem, koiter, filer_interpolation, solver, other]

    args = get_args(settings)

    return args


def paropt_options(args):
    options = {
        "algorithm": "mma",
        "mma_max_iterations": args.maxiter,
        "mma_init_asymptote_offset": 0.2,
        "max_major_iters": 100,
        "mma_move_limit": 0.2,
        "penalty_gamma": 1e3,
        "qn_subspace_size": 10,
        "qn_type": "bfgs",
        "abs_res_tol": 1e-8,
        "starting_point_strategy": "affine_step",
        "barrier_strategy": "mehrotra_predictor_corrector",
        "barrier_strategy": "mehrotra",
        "use_line_search": False,
        "output_file": os.path.join(args.prefix, "paropt.out"),
        "mma_output_file": os.path.join(args.prefix, "paropt.mma"),
    }
    return options


def create_topo_model(args):
    topo = make_model(
        nx=args.nx,
        ny=int(args.yxratio * args.nx),
        Ly=args.yxratio * 1.0,
        rfact=args.r,
        N=args.N,
        solver_type=args.solver_type,
        adjoint_method=args.adjoint_method,
        adjoint_options=args.adjoint_options,
        ptype_K=args.ptype_K,
        ptype_M=args.ptype_M,
        ptype_G=args.ptype_G,
        rho0_K=args.rho0_K,
        rho0_M=args.rho0_M,
        rho0_G=args.rho0_G,
        p=args.p0,
        q=args.q0,
        projection=args.projection,
        b0=args.b0,
        rtol=args.rtol,
        eig_atol=args.eig_atol,
        sigma=args.sigma,
        domain=args.domain,
    )
    return topo


def set_node(topo, args):

    args.node = []
    args.index = []

    if args.objf == "aggregate" or "aggregate" in args.confs:
        m = args.nx
        n = int(args.yxratio * args.nx)
        args.index.append(int((m // 2 + 1) * (n + 1) - 1))  # top middle point
        # args.index.append(int((m // 2) * (n + 1) + n//2))  # center-point
        args.node.append(2 * args.index)  # along x-axis

    if args.objf == "aggregate-max" or "aggregate-max" in args.confs:
        for i in range(topo.nnodes):
            args.node.append(i)

    args.node = np.unique(args.node)
    return


def check_gradients(opt, args):
    if args.grad_check:
        problem = ParOptProb(MPI.COMM_SELF, opt, args, grad_check=True)
        for dh in [1e-6]:
            problem.checkGradients(dh)
            print("\n")
    return


def get_topo():
    args = settings()
    model = create_topo_model(args)
    opt = BucklingOpt(model, args.ks_rho, args.rho_agg)
    problem = ParOptProb(MPI.COMM_SELF, opt, args)
    return opt.topo, problem


if __name__ == "__main__":
    # Initialize settings
    args = settings()

    import kokkos

    kokkos.initialize()

    # Create topology optimization model
    topo = create_topo_model(args)
    set_node(topo, args)
    opt = BucklingOpt(topo, args.ks_rho, args.rho_agg, args.node)

    # Check the gradients
    check_gradients(opt, args)

    # Create the ParOpt problem
    problem = ParOptProb(MPI.COMM_SELF, opt, args)
    options = paropt_options(args)

    # Set up the optimizer
    optimizer = ParOpt.Optimizer(problem, options)
    optimizer.optimize()

    kokkos.finalize()

    print("=====================")
    print("Optimization complete")

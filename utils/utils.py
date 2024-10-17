import argparse
from collections import OrderedDict
from datetime import datetime
import json
import logging
import os
from shutil import rmtree
import sys
import time
from time import perf_counter_ns

from icecream import ic, install
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from plot_history import plot_history

try:

    import matplotlib
    import matplotlib.font_manager
    import niceplots

    logging.getLogger("matplotlib.font_manager").disabled = True
    plt.style.use(niceplots.get_style())
    matplotlib.rcParams.update(
        {
            "font.sans-serif": "Arial",
            "font.family": "sans-serif",
        }
    )
except:
    pass

coolwarm = plt.colormaps["coolwarm"]

niceColors = dict()
niceColors["Blue"] = coolwarm(0.0)
niceColors["Red"] = coolwarm(1.0)
niceColors["Orange"] = "#e29400ff"
niceColors["Green"] = "#00a650ff"
colors = list(niceColors.values())


def line(msg=""):
    print(f"Debug {sys._getframe().f_back.f_lineno}: {msg}")


class Log:
    log_name = "stdout.log"

    @staticmethod
    def set_log_path(log_path):
        Log.log_name = log_path

    @staticmethod
    def log(txt="", end="\n"):
        with open(Log.log_name, "a") as f:
            f.write(txt + end)
        return


class Logger:

    def __init__(self, paropt, grad_check=False, problem=None):
        self.paropt = paropt
        self.prob = paropt.prob
        self.args = paropt.args

        self.problem = problem
        self.grad_check = grad_check

        self.iter = 0
        self.draw_every = 5
        self.profile = paropt.prob.topo.profile

        self.foi = OrderedDict({"volume": "n/a"})

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    def initialize_output(self):
        if not self.grad_check:
            print("\nOptimization started")
            print("====================")

            # print the objective function and constraints
            if self.problem == "natural frequency":
                self.ny = self.args.ny
                self.nx = self.prob.topo.conn.shape[0] // self.ny
            else:
                self.nx = self.args.nx
                self.ny = self.prob.topo.conn.shape[0] // self.nx

            print(f"Problem: {self.problem}")
            print(f"Objective: {self.args.objf}")
            print(f"Constraints: {self.args.confs}")
            print(f"Number of elements: {self.nx} x {self.ny}")
            print(f"Number of design variables: {self.prob.topo.nvars}")
            print(f"Sigma0: {self.args.sigma}")
            print(f"Sigma scale: {self.args.sigma_scale}\n")

            self.create_folder()
            Log.set_log_path(os.path.join(self.args.prefix, "stdout.log"))
            timer_set_log_path(os.path.join(self.args.prefix, "profiler.log"))

            with open(os.path.join(self.args.prefix, "options.txt"), "w") as f:
                f.write("Options:\n")
                for k, v in vars(self.args).items():
                    f.write(f"{k:<20}{v}\n")

            if self.args.objf == "temperature" or self.args.objf == "aggregate":
                self.plot_heat_source()
        return

    def write_output(self):
        if not self.grad_check:
            if self.problem == "buckling":
                self.write_output_buckling()
            elif self.problem == "thermal":
                self.write_output_thermal()
            else:
                self.write_output_natural_frequency()
            t1 = time.time()
            t = t1 - self.t0
            # check if the time is in seconds or minutes or hours
            if t < 60:
                print("Total time: %16.2f s" % t)
            elif t < 3600:
                print("Total time: %16.2f m" % (t / 60))
            else:
                print("Total time: %16.2f h" % (t / 3600))

            plt.close("all")

            try:
                import psutil

                process = psutil.Process(os.getpid())
                mem = process.memory_info().rss / 1024 / 1024
                print("Memory usage: %14.2f MB" % mem)
            except:
                pass

            print(
                "==============================end",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                flush=True,
            )

            self.iter += 1

        return

    def write_output_natural_frequency(self):
        self.write_stdout()
        self.write_adjoint()

        if self.iter % self.draw_every == 0:
            self.write_design_natrual_frequency()

        if self.iter % (10 * self.draw_every) == 0 and self.iter != 0:
            self.write_optimization()

        if self.iter % (10 * self.draw_every) == 0:
            self.write_eigenvectors(x=8, y=4)
            self.write_vtk_natrual_frequency()

        if self.iter % (4 * self.draw_every) == 0:
            self.write_residuals()
            self.write_time()

        return

    def write_output_thermal(self):
        self.write_stdout()
        self.write_adjoint()

        if self.iter % self.draw_every == 0:
            self.write_design_thermal()

        if self.iter % (10 * self.draw_every) == 0 and self.iter != 0:
            self.write_optimization()

        if self.iter % (10 * self.draw_every) == 0:
            self.write_temperatures()
            self.write_eigenvectors(x=5, y=5)
            self.write_vtk_thermal()

        if self.iter % (4 * self.draw_every) == 0:
            self.write_residuals()
            self.write_time()

        return

    def write_output_buckling(self):
        self.write_stdout()
        self.write_adjoint() if "aggregate" in self.args.confs else None

        if self.iter % self.draw_every == 0:
            self.write_design_buckling()

        if self.iter % (10 * self.draw_every) == 0 and self.iter != 0:
            self.write_optimization()

        if self.iter % (4 * self.draw_every) == 0:
            self.write_eigenvectors(x=1, y=1)
            self.write_vtk_natrual_frequency()

        if self.iter % (4 * self.draw_every) == 0:
            if (
                "aggregate" in self.args.confs
                and self.profile["adjoint_method"] != "approx-lanczos"
            ):
                self.write_residuals()

    def set_interpolation_parameters(self):
        if not self.grad_check:
            print(f"\nIteration: {self.iter}")
            self.t0 = time.time()
            if self.iter > self.args.psiter:
                if self.iter % self.args.ppiter == 0:
                    self.prob.topo.p += self.args.pp
                    self.prob.topo.p = round(min(self.prob.topo.p, self.args.pmax), 2)
            if self.prob.topo.fltr.projection and self.iter > self.args.bsiter:
                if self.iter % self.args.bpiter == 0:
                    self.prob.topo.fltr.beta += self.args.bp
                    self.prob.topo.fltr.beta = round(
                        min(self.prob.topo.fltr.beta, self.args.bmax), 2
                    )
            if self.args.pp > 0:
                print(f"p: {self.prob.topo.p}")
            if self.args.projection:
                print(f"beta: {self.prob.topo.fltr.beta}")

        return

    def write_design_natrual_frequency(self):
        des_dir = os.path.join(self.args.prefix, "design")
        os.makedirs(des_dir, exist_ok=True)
        des_path = os.path.join(des_dir, "%d.png" % self.iter)
        self.prob.topo.plot_design(path=des_path, node_sets=True)
        return

    def write_design_buckling(self):
        des_dir = os.path.join(self.args.prefix, "design")
        os.makedirs(des_dir, exist_ok=True)
        des_path = os.path.join(des_dir, "%d.png" % self.iter)
        self.prob.topo.plot_design(path=des_path, index=self.args.index)
        return

    def write_design_thermal(self):
        des_dir = os.path.join(self.args.prefix, "design")
        os.makedirs(des_dir, exist_ok=True)
        des_path = os.path.join(des_dir, "%d.png" % self.iter)

        if self.args.objf == "aggregate":
            node = self.args.node
        else:
            node = None

        set1, set2 = [], []
        for case_name in self.prob.cases:
            for key, value in self.prob.heat_func[case_name].items():
                if value(1) > 0:
                    set1.append(key)  # heat-up region
                elif value(1) < 0:
                    set2.append(key)  # cool-down region
            self.prob.topo.plot_design(set1, set2, path=des_path, node=node)
        return

    def write_temperatures(self):
        temp_dir = os.path.join(self.args.prefix, "temperature")
        os.makedirs(temp_dir, exist_ok=True)

        # add the temperatures in a json file
        if self.iter == 0:
            temp = {}
        else:
            with open(os.path.join(temp_dir, "temperature.json"), "r") as f:
                temp = json.load(f)

        for case_name in self.prob.cases:
            n = 1
            temp_modal_path = lambda i: os.path.join(
                temp_dir, "%d_modal_%d.png" % (self.iter, i)
            )
            u_modal = self.prob.plot_temperature_history(
                case_name,
                hist="modal",
                path=temp_modal_path,
                skip=self.prob.nsteps // n,
            )
            # temp_full_path = lambda i: os.path.join(
            #     temp_dir, "%d_full_%d.png" % (self.iter, i)
            # )
            # u_full = self.prob.plot_temperature_history(
            #     case_name,
            #     hist="full",
            #     path=temp_full_path,
            #     skip=self.prob.nsteps // n,
            # )

        temp[self.iter] = {"modal": u_modal.tolist()}

        with open(os.path.join(temp_dir, "temperature.json"), "w") as f:
            f.write(json.dumps(temp) + "\n")

        return

    def plot_heat_source(self):
        t = np.linspace(0, self.args.heat_funcs["tfinal"], 100)

        for case_name in self.args.cases:
            heat_dir = os.path.join(self.args.prefix, "heat_source.png")

            fig, ax = plt.subplots()
            for key, value in self.prob.heat_func[case_name].items():
                ax.plot(t, [value(ti) for ti in t], label=key)
            plt.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Temperature")
            plt.savefig(heat_dir, dpi=300)
            plt.close(fig)
        return

    def write_residuals(self):
        res_dir = os.path.join(self.args.prefix, "adjoint_residuals")
        os.makedirs(res_dir, exist_ok=True)
        res_path = os.path.join(res_dir, "%d.png" % self.iter)
        self.prob.topo.plot_residuals(res_path)

        # add the residuals in a json file
        if self.iter == 0:
            res = {}
        else:
            with open(os.path.join(res_dir, "residuals.json"), "r") as f:
                res = json.load(f)

        res[self.iter] = self.profile["adjoint residuals"]
        with open(os.path.join(res_dir, "residuals.json"), "w") as f:
            f.write(json.dumps(res) + "\n")

        return

    def write_vtk_thermal(self):
        vtk_dir = os.path.join(self.args.prefix, "vtk")
        os.makedirs(vtk_dir, exist_ok=True)
        vtk_path = os.path.join(vtk_dir, "it_%d.vtk" % self.iter)
        vtk_nodal_sols = {}
        vtk_nodal_vecs = {}

        # Assign the nodal solutions
        xfull = self.prob.topo.x[self.prob.topo.fltr.dvmap]
        vtk_nodal_sols["design"] = xfull
        vtk_nodal_sols["rho"] = self.prob.topo.rho[:]

        to_vtk(
            vtk_path,
            self.prob.topo.conn,
            self.prob.topo.X,
            nodal_sols=vtk_nodal_sols,
        )
        return

    def write_vtk_natrual_frequency(self):
        vtk_dir = os.path.join(self.args.prefix, "vtk")
        os.makedirs(vtk_dir, exist_ok=True)
        vtk_path = os.path.join(vtk_dir, "it_%d.vtk" % self.iter)
        vtk_nodal_sols = {}
        vtk_nodal_vecs = {}

        # Assign the nodal solutions
        if self.prob.topo.fltr.dvmap is not None:
            xfull = self.prob.topo.x[self.prob.topo.fltr.dvmap]
        else:
            xfull = self.prob.topo.x

        vtk_nodal_sols["design"] = xfull
        vtk_nodal_sols["rho"] = self.prob.topo.rho[:]
        for i in range(self.prob.topo.N):
            vtk_nodal_vecs["phi%d" % i] = [
                self.prob.topo.Q[0::2, i],
                self.prob.topo.Q[1::2, i],
            ]

        to_vtk(
            vtk_path,
            self.prob.topo.conn,
            self.prob.topo.X,
            nodal_sols=vtk_nodal_sols,
            nodal_vecs=vtk_nodal_vecs,
        )
        return

    def write_stdout(self, full_output=True):
        # Print problem overview
        if self.iter == 0:
            if self.args.case_num > 0:
                Log.log(f"=== Case {self.args.case_num} ===")
            else:
                Log.log("=== Problem overview ===")
            Log.log(f"objective:   {self.args.objf}")
            Log.log(f"constraints: {self.args.confs}")
            Log.log(f"num of elements: {self.nx} x {self.ny}")
            Log.log(f"num of dof:  {self.prob.topo.nvars}\n")

        if self.args.objf == "frequency":
            self.foi["lam"] = self.prob.topo.lam[0]
        self.foi["volume"] = self.paropt.vol_frac

        # if "compliance" in self.args.confs:
        #     self.foi["c"] = self.paropt.c

        # if "ks-buckling" in self.args.confs:
        #     self.foi["ks"] = self.paropt.BLF_ks

        if (
            self.args.objf == "ks-buckling"
            or self.args.objf == "koiter-ks-lams"
            or self.args.objf == "koiter-ks-lams-b"
        ):
            self.foi["BLF0"] = self.paropt.prob.topo.BLF[0]

        if "aggregate" in self.args.confs:
            self.foi["aggregate"] = self.paropt.h

        if (
            self.args.objf == "compliance-buckling"
            or self.args.objf == "koiter-ks-lams-bc"
        ):
            self.foi["c/c0"] = self.paropt.c_norm
            self.foi["ks/ks0"] = self.paropt.ks_norm
            self.foi["BLF0"] = self.prob.topo.BLF[0]

        teig = self.profile["eigenvalue solve time"]
        tkoiter = self.profile["koiter time"]
        tderiv = self.profile["total derivative time"]

        # Log function values and time
        if self.iter % 10 == 0:
            Log.log("\n%4s%15s" % ("iter", "obj"), end="")

            for k in self.foi.keys():
                Log.log("%10s" % k, end="")

            Log.log(
                "%10s%10s%10s%10s" % ("1e-4", "1e-3", "1e-2", "1e-1"),
                end="",
            )

            Log.log("%13s%13s" % ("a", "b"), end="")
            # Log.log("%10s%10s%10s" % ("eig", "dev", "koiter"))
            Log.log("")

        Log.log("%4d%15.5e" % (self.iter, self.paropt.obj), end="")

        for v in self.foi.values():
            if not isinstance(v, str):
                Log.log("%10.3f" % v, end="")

        xib = [1e-4, 1e-3, 1e-2, 1e-1]

        for xi in xib:
            xi *= np.linalg.norm(self.prob.topo.Q0)

            if (
                self.args.objf
                in {"koiter-ks-lams-b", "koiter-ks-lams-bc", "compliance-buckling"}
                or "koiter-b" in self.args.confs
            ):
                lam_s = self.prob.topo.get_lams_b(
                    self.prob.topo.lam[0], self.prob.topo.b, xi
                )
                if lam_s < 1e-5:
                    lam_s = 0
            else:
                aa = np.abs(self.prob.topo.a * xi)
                lam_s = self.prob.topo.lam[0] * (1 + 2 * aa - 2 * np.sqrt(aa + aa**2))

            Log.log("%10.3f" % lam_s, end="")

        Log.log("%13.3e%13.3e" % (self.prob.topo.a, self.prob.topo.b), end="")
        Log.log("")

        # Log.log("%10.3f%10.3f%10.3f" % (teig, tderiv, tkoiter))

        return

    def read_stdout(self, dir=None):
        # read the file and skip the first 6 lines
        if dir is None:
            dir = os.path.join(self.args.prefix, "stdout.log")
        df = pd.read_csv(dir, sep=r"\s+", skiprows=6, header=None)

        # skip the lines with "iter"
        df = df.drop(df[df.iloc[:, 0].str.contains("iter")].index)
        df.reset_index(drop=True, inplace=True)

        # convert to float
        for i in range(1, df.shape[1]):
            df[i] = df[i].astype(float)

        return df

    def write_time(self):
        df = self.read_stdout()
        teig = df.iloc[:, -3].values
        tsolve = df.iloc[:, -2].values
        nfeig = df.iloc[:, -5].values
        nfsol = df.iloc[:, -4].values

        fig, axs = plt.subplots(figsize=(12, 5), constrained_layout=True)

        title = self.profile["adjoint_method"]

        if self.profile["adjoint_method"] == "shift-invert":
            if self.profile["adjoint_options"]["update_guess"] is True:
                title = f"SIBK"
            else:
                title = f"SIBK-NU"

        if self.profile["adjoint_options"]["lanczos_guess"] is True:
            title = title + "-LAA"

        axs.plot(range(self.iter + 1), teig, label="teig")
        axs.plot(range(self.iter + 1), tsolve, label="tsolve")

        axs2 = axs.twinx()
        l1 = axs2.plot(range(self.iter + 1), tsolve / teig, label="t-ratio", c="r")
        axs2.axhline(
            y=np.mean(tsolve / teig), c=l1[0].get_color(), linestyle="--", alpha=0.5
        )
        axs2.text(
            self.iter,
            np.mean(tsolve / teig),
            f"{np.mean(tsolve / teig):.2f}",
            color="gray",
        )

        l2 = axs2.plot(
            range(self.iter + 1), nfsol / nfeig, label="num factor-ratio", c="g"
        )
        axs2.axhline(
            y=np.mean(nfsol / nfeig), c=l2[0].get_color(), linestyle="--", alpha=0.5
        )
        axs2.text(
            self.iter,
            np.mean(nfsol / nfeig),
            f"{np.mean(nfsol / nfeig):.2f}",
            color="gray",
        )

        axs2.set_ylabel("time ratio")
        axs2.spines["right"].set_visible(True)
        axs2.set_ylim(ymin=0)

        axs.set_xlabel("Iteration")
        axs.set_ylabel("time (s)")
        axs.set_title(title)

        handles, labels = axs.get_legend_handles_labels()
        handles2, labels2 = axs2.get_legend_handles_labels()
        axs.legend(
            handles + handles2,
            labels + labels2,
            loc="center left",
            bbox_to_anchor=(1.1, 0.5),
        )

        plt.savefig(os.path.join(self.args.prefix, "time.png"), dpi=300)
        plt.close(fig)

        return

    def write_adjoint(self):
        with open(os.path.join(self.args.prefix, "adjoint.log"), "a") as f:
            self.prob.topo.add_check_adjoint_residual()
            f.write("Iteration: %d \n" % self.iter)
            for i in range(self.prob.topo.N):
                f.write(
                    "||Adjoint norm[%2d]||: %15.8e  Ortho: %15.8e  lam: %15.15f\n"
                    % (
                        i,
                        self.profile["adjoint norm[%2d]" % i],
                        self.profile["adjoint ortho[%2d]" % i],
                        self.profile["adjoint lam[%2d]" % i],
                    )
                )
            f.write("\n")

        return

    def write_eigenvectors(self, x=5, y=5):
        eig_dir = os.path.join(self.args.prefix, "eigenvectors")
        os.makedirs(eig_dir, exist_ok=True)
        eig_path = os.path.join(eig_dir, "%d.png" % self.iter)
        N = self.prob.topo.Q.shape[1]

        ny = int(np.sqrt(N))
        nx = N // ny
        if nx * ny < N:
            nx += 1

        fig, ax = plt.subplots(ny, nx, figsize=(x, y), constrained_layout=True)

        if ny == 1 and nx == 1:
            ax = np.array([[ax]])

        for j in range(ny):
            for i in range(nx):
                k = i + nx * j
                if k < N:
                    self.prob.topo.plot_mode(k, ax[j, i])
                else:
                    ax[j, i].axis("off")

        fig.savefig(eig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        return

    def write_optimization(self):
        paropt_out = os.path.join(self.args.prefix, "paropt.out")
        paropt_mma = os.path.join(self.args.prefix, "paropt.mma")
        paropt_out_img = os.path.join(self.args.prefix, "paropt_out")
        paropt_mma_img = os.path.join(self.args.prefix, "paropt_mma")
        plot_history(paropt_out, paropt_out_img)
        plot_history(paropt_mma, paropt_mma_img)

        return

    def create_folder(self):
        os.makedirs(self.args.prefix, exist_ok=True)

        name = f"{self.args.domain}"
        if not os.path.isdir(os.path.join(self.args.prefix, name)):
            os.mkdir(os.path.join(self.args.prefix, name))
        self.args.prefix = os.path.join(self.args.prefix, name)

        # make a folder inside each domain folder to store the results of each run
        name2 = f"{self.args.objf}{self.args.confs}"
        if not os.path.isdir(os.path.join(self.args.prefix, name2)):
            os.mkdir(os.path.join(self.args.prefix, name2))
        self.args.prefix = os.path.join(self.args.prefix, name2)

        if "nx" in self.args:
            n = f"{self.args.nx}"
        elif "ny" in self.args:
            n = f"{self.args.ny}"
        self.args.prefix = os.path.join(self.args.prefix, "n=" + n)

        # if self.args.confs != []:
        #     v = f"{self.args.vol_frac_ub:.2f}"
        #     self.args.prefix = self.args.prefix + ", v=" + v

        # r = f"{self.args.r}"
        # self.args.prefix = self.args.prefix + ", r=" + r

        # e = f"{self.args.solver_type}"
        # self.args.prefix = self.args.prefix + ", " + e

        # s = f"{self.args.adjoint_method}"
        # self.args.prefix = self.args.prefix + ", " + s

        N = f"{self.args.N}"
        self.args.prefix = self.args.prefix + ", N=" + N

        # if self.args.adjoint_method == "shift-invert":
        #     if self.args.adjoint_options["update_guess"]:
        #         self.args.prefix = self.args.prefix + ", ug=1"
        #     else:
        #         self.args.prefix = self.args.prefix + ", ug=0"

        #     self.args.prefix = self.args.prefix + ", bs=" + str(self.args.adjoint_options["bs_target"])

        # if "Mx" in self.args:
        #     self.args.prefix = self.args.prefix + ", Mx=" + str(self.args.Mx)

        # if "My" in self.args:
        #     self.args.prefix = self.args.prefix + ", My=" + str(self.args.My)

        # if "ns" in self.args:
        #     self.args.prefix = self.args.prefix + ", ns=" + str(self.args.ns)

        # if self.args.projection:
        #     self.args.prefix = self.args.prefix + ", bp=" + str(self.args.bp)

        # if self.args.pp > 0:
        #     self.args.prefix = self.args.prefix + ", pp=" + str(self.args.pp)

        # if "sigma_scale" in self.args:
        #     self.args.prefix = self.args.prefix + ", sc=" + str(self.args.sigma_scale)

        # if "sigma" in self.args:
        #     self.args.prefix = self.args.prefix + ", s=" + str(self.args.sigma)

        # if "h_ub" in self.args:
        #     self.args.prefix = self.args.prefix + ", h=" + str(self.args.h_ub)

        # if "sigma" in self.args:
        #     self.args.prefix = self.args.prefix + ", s=" + str(self.args.sigma)

        # if "sigma_scale" in self.args:
        #     self.args.prefix = self.args.prefix + ", sc=" + str(self.args.sigma_scale)

        # use secific format for xi
        if "xi" in self.args and self.args.objf != "ks-buckling":
            self.args.prefix = self.args.prefix + ", xi=" + f"{self.args.xi:.0e}"

        # # if args have kappa, add it to the prefix
        # if "kappa" in self.args:
        #     self.args.prefix = self.args.prefix + ", k=" + str(self.args.kappa)

        # if "beta" in self.args:
        #     self.args.prefix = self.args.prefix + ", " + str(self.args.beta)

        if self.args.note != "":
            self.args.prefix = self.args.prefix + ", " + self.args.note

        if self.args.case_num > 0:
            self.args.prefix = self.args.prefix + " (" + str(self.args.case_num) + ")"

        if not os.path.isdir(self.args.prefix):
            os.mkdir(self.args.prefix)
        else:
            rmtree(self.args.prefix)
            os.mkdir(self.args.prefix)

        # if self.args.note != "":
        #     self.args.prefix = os.path.join(self.args.prefix, self.args.note)
        #     self.args.prefix = self.args.prefix + ", " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # else:
        #     self.args.prefix = os.path.join(
        #         self.args.prefix, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #     )
        # os.mkdir(self.args.prefix)

        return


class MyProfiler:
    counter = 0  # a static variable
    timer_is_on = True
    print_to_stdout = False
    buffer = []
    istart = []  # stack of indices of open parantheses
    pairs = {}
    t_min = 1  # unit: ms
    log_name = "profiler.log"
    old_log_removed = False
    saved_times = {}

    @staticmethod
    def timer_set_log_path(log_path):
        MyProfiler.log_name = log_path

    @staticmethod
    def timer_set_threshold(t: float):
        """
        Don't show entries with elapse time smaller than this. Unit: ms
        """
        MyProfiler.t_min = t
        return

    @staticmethod
    def timer_to_stdout():
        """
        print the profiler output to stdout, otherwise save it as a file
        """
        MyProfiler.print_to_stdout = True
        return

    @staticmethod
    def timer_on():
        """
        Call this function before execution to switch on the profiler
        """
        MyProfiler.timer_is_on = True
        return

    @staticmethod
    def timer_off():
        """
        Call this function before execution to switch off the profiler
        """
        MyProfiler.timer_is_on = False
        return

    @staticmethod
    def time_this(func):
        """
        Decorator: time the execution of a function
        """
        tab = "    "
        fun_name = func.__qualname__

        if not MyProfiler.timer_is_on:

            def wrapper(*args, **kwargs):
                ret = func(*args, **kwargs)
                return ret

            return wrapper

        def wrapper(*args, **kwargs):
            info_str = f"{tab*MyProfiler.counter}{fun_name}() called"
            entry = {"msg": f"{info_str:<40s}", "type": "("}
            MyProfiler.buffer.append(entry)

            MyProfiler.counter += 1
            t0 = perf_counter_ns()
            ret = func(*args, **kwargs)
            t1 = perf_counter_ns()
            t_elapse = (t1 - t0) / 1e6  # unit: ms
            MyProfiler.counter -= 1

            info_str = f"{tab*MyProfiler.counter}{fun_name}() return"
            entry = {
                "msg": f"{info_str:<80s} ({t_elapse:.2f} ms)",
                "type": ")",
                "fun_name": fun_name,
                "t": t_elapse,
            }
            MyProfiler.buffer.append(entry)

            # Once the most outer function returns, we fltr the buffer such
            # that we only keep entry pairs whose elapse time is above threshold
            if MyProfiler.counter == 0:
                for idx, entry in enumerate(MyProfiler.buffer):
                    if entry["type"] == "(":
                        MyProfiler.istart.append(idx)
                    if entry["type"] == ")":
                        try:
                            start_idx = MyProfiler.istart.pop()
                            if entry["t"] > MyProfiler.t_min:
                                MyProfiler.pairs[start_idx] = idx
                        except IndexError:
                            print("[Warning]Too many return message")

                # Now our stack should be empty, otherwise we have unpaired
                # called/return message
                if MyProfiler.istart:
                    print("[Warning]Too many called message")

                # Now, we only keep the entries for expensive function calls
                idx = list(MyProfiler.pairs.keys()) + list(MyProfiler.pairs.values())
                if idx:
                    idx.sort()
                keep_buffer = [MyProfiler.buffer[i] for i in idx]

                if MyProfiler.print_to_stdout:
                    for entry in keep_buffer:
                        print(entry["msg"])
                else:
                    if (
                        os.path.exists(MyProfiler.log_name)
                        and not MyProfiler.old_log_removed
                    ):
                        os.remove(MyProfiler.log_name)
                        MyProfiler.old_log_removed = True
                    with open(MyProfiler.log_name, "a") as f:
                        for entry in keep_buffer:
                            f.write(entry["msg"] + "\n")

                # Save time information to dictionary
                for entry in keep_buffer:
                    if "t" in entry.keys():
                        _fun_name = entry["fun_name"]
                        _t = entry["t"]
                        if _fun_name in MyProfiler.saved_times.keys():
                            MyProfiler.saved_times[_fun_name].append(_t)
                        else:
                            MyProfiler.saved_times[_fun_name] = [_t]

                # Reset buffer and pairs
                MyProfiler.buffer = []
                MyProfiler.pairs = {}
            return ret

        return wrapper


def get_args(settings):
    parser = argparse.ArgumentParser()

    for d in settings:
        for key, v in d.items():
            key = key.replace("_", "-")
            if isinstance(v, list):
                parser.add_argument(f"--{key}", type=type(v[0]), nargs="*", default=v)
            elif isinstance(v, bool):
                parser.add_argument(f"--{key}", action="store_true", default=v)
            elif isinstance(v, dict):
                for k, w in v.items():
                    k = k.replace("_", "-")
                    parser.add_argument(f"--{k}", type=int, default=w)
            else:
                parser.add_argument(f"--{key}", type=type(v), default=v)

    args = parser.parse_args()

    adjoint_options = {}
    for key, value in args._get_kwargs():
        if key in ["lanczos_guess"]:
            if value == 1:
                value = True
            elif value == 0:
                value = False
            adjoint_options[key] = value

        elif key in ["update_guess"] and args.adjoint_method == "shift-invert":
            if value == 1:
                value = True
            elif value == 0:
                value = False
            adjoint_options[key] = value

        elif key in ["bs_target"] and args.adjoint_method == "shift-invert":
            adjoint_options[key] = int(value)

    args.adjoint_options = adjoint_options

    return args


time_this = MyProfiler.time_this
timer_on = MyProfiler.timer_on
timer_off = MyProfiler.timer_off
timer_to_stdout = MyProfiler.timer_to_stdout
timer_set_threshold = MyProfiler.timer_set_threshold
timer_set_log_path = MyProfiler.timer_set_log_path


def to_vtk(vtk_path, conn, X, nodal_sols={}, cell_sols={}, nodal_vecs={}, cell_vecs={}):
    """
    Generate a vtk given conn, X, and optionally list of nodal solutions

    Args:
        nodal_sols: dictionary of arrays of length nnodes
        cell_sols: dictionary of arrays of length nelems
        nodal_vecs: dictionary of list of components [vx, vy], each has length nnodes
        cell_vecs: dictionary of list of components [vx, vy], each has length nelems
    """
    # vtk requires a 3-dimensional data point
    X = np.append(X, np.zeros((X.shape[0], 1)), axis=1)

    nnodes = X.shape[0]
    nelems = conn.shape[0]

    # Create a empty vtk file and write headers
    with open(vtk_path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write("my example\n")
        fh.write("ASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")

        # Write nodal points
        fh.write("POINTS {:d} double\n".format(nnodes))
        for x in X:
            row = f"{x}"[1:-1]  # Remove square brackets in the string
            fh.write(f"{row}\n")

        # Write connectivity
        size = 5 * nelems

        fh.write(f"CELLS {nelems} {size}\n")
        for c in conn:
            node_idx = f"{c}"[1:-1]  # remove square bracket [ and ]
            npts = 4
            fh.write(f"{npts} {node_idx}\n")

        # Write cell type
        fh.write(f"CELL_TYPES {nelems}\n")
        for c in conn:
            vtk_type = 9
            fh.write(f"{vtk_type}\n")

        # Write solution
        if nodal_sols or nodal_vecs:
            fh.write(f"POINT_DATA {nnodes}\n")

        if nodal_sols:
            for name, data in nodal_sols.items():
                fh.write(f"SCALARS {name} double 1\n")
                fh.write("LOOKUP_TABLE default\n")
                for val in data:
                    fh.write(f"{val}\n")

        if nodal_vecs:
            for name, data in nodal_vecs.items():
                fh.write(f"VECTORS {name} double\n")
                for val in np.array(data).T:
                    fh.write(f"{val[0]} {val[1]} 0.\n")

        if cell_sols or cell_vecs:
            fh.write(f"CELL_DATA {nelems}\n")

        if cell_sols:
            for name, data in cell_sols.items():
                fh.write(f"SCALARS {name} double 1\n")
                fh.write("LOOKUP_TABLE default\n")
                for val in data:
                    fh.write(f"{val}\n")

        if cell_vecs:
            for name, data in cell_vecs.items():
                fh.write(f"VECTORS {name} double\n")
                for val in np.array(data).T:
                    fh.write(f"{val[0]} {val[1]} 0.\n")

    return

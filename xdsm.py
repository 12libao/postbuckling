import matplotlib
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from pyxdsm.XDSM import (
    DOE,
    FUNC,
    GROUP,
    IFUNC,
    IGROUP,
    LEFT,
    METAMODEL,
    OPT,
    RIGHT,
    SOLVER,
    SUBOPT,
    XDSM,
)

plt.rc("text", usetex=True)
# use sansmath package and bm package
matplotlib.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{sansmath}\usepackage{amsmath}",
    }
)


# Purpose: Create an XDSM diagram for the Koiter Asymptotic Theory
x = XDSM(
    auto_fade={
        # "inputs": "none",
        "outputs": "connected",
        "connections": "outgoing",
        # "processes": "none",
    }
)

x.add_system("K0", FUNC, r"\text{Assemble} $\\$ K_0")
x.add_system(
    "Equilibrium",
    SOLVER,
    r"\text{Solve:} $\\$ \text{Pre-Buckling System} $\\$ $\\$  K_0 u_0 = f",
)
x.add_system("G0", FUNC, r"\text{Assemble} $\\$ G_0")
x.add_system(
    "Linear Buckling",
    SOLVER,
    r"\text{Solve:} $\\$ \text{Linear Buckling System} $\\$ $\\$ (K_0 + \lambda_c G_0) \phi_1 = 0",
)
x.add_system("K1", FUNC, r"\text{Assemble} $\\$ K_1, K_{11}, G_1")
x.add_system(
    "Koiter-a",
    DOE,
    r"\text{Koiter Asymptotic Theory:} $\\$  $\\$ a = \frac{3}{2} \phi_1^T K_1 \phi_1",
)
x.add_system(
    "Post Buckling",
    SOLVER,
    r"\text{Solve:} $\\$ \text{Post-Buckling System for } u_2 $\\$  $\\$ \begin{bmatrix} G_0 - \mu K_0 &K_0 u_1 \\ u_1^T K_0 & 0 \end{bmatrix} \begin{bmatrix} u_2 \\ \nu \end{bmatrix} = \begin{bmatrix} \mu (\frac{1}{2} K_1 + G_1) u_1 \\ 0 \end{bmatrix}",
)
x.add_system(
    "Koiter-b",
    DOE,
    r"\text{Koiter Asymptotic Theory:} $\\$  $\\$ b = \|\phi_1\|^2 \left(u_2^T K_1 u_1 + \frac{1}{2} u_1^T K_{11} u_1 + 2 u_1^T K_1 u_2\right)",
)

x.connect("K0", "Equilibrium", "K_0")
x.connect("Equilibrium", "G0", "u_0")
x.connect("G0", "Linear Buckling", "G_0")
x.connect("K0", "Linear Buckling", "K_0")
x.connect("Linear Buckling", "K1", r"\phi_1")
x.connect("K1", "Koiter-a", "K_1")

x.connect("K0", "Post Buckling", "K_0")
x.connect("G0", "Post Buckling", "G_0")
x.connect("Linear Buckling", "Post Buckling", r"\lambda_c, \phi_1")
x.connect("K1", "Post Buckling", "K_1, G_1")

x.connect("Linear Buckling", "Koiter-a", r"\phi_1")
x.connect("Linear Buckling", "Koiter-b", r"\phi_1")
x.connect("Post Buckling", "Koiter-b", "u_2")
x.connect("K1", "Koiter-b", "K_1, K_{11}")


x.add_input("K0", r"x")
x.add_input("Equilibrium", r"f")
x.add_input("G0", r"x")
x.add_input("K1", r"x")

x.add_output("Linear Buckling", r"\lambda_c, \phi_1, u_1", side=RIGHT)
x.add_output("Koiter-a", r"a", side=RIGHT)
x.add_output("Post Buckling", r"\nu, u_2", side=RIGHT)
x.add_output("Koiter-b", r"b", side=RIGHT)

x.add_process(["K0", "Equilibrium", "G0", "Linear Buckling"], arrow=True)
x.add_process(["Linear Buckling", "K1", "Koiter-a", "right_output_Koiter-a"], arrow=True)
x.add_process(["K1", "Post Buckling", "Koiter-b", "right_output_Koiter-b"], arrow=True)
# x.add_process(["Linear Buckling", "left_output_Linear Buckling"], arrow=True)
x.add_process(["Linear Buckling", "right_output_Linear Buckling"], arrow=True)
x.add_process(["Post Buckling", "right_output_Post Buckling"], arrow=True)

x.write("xdsm", cleanup=False, quiet=True)
x.write_sys_specs("sink_specs")

images = convert_from_path("xdsm.pdf", dpi=600)
images[0].save("./output/koiter_xdsm.png")


##################################################
x = XDSM(
    auto_fade={
        # "inputs": "none",
        "outputs": "connected",
        "connections": "outgoing",
        # "processes": "none",
    }
)

x.add_system(
    "R1",
    FUNC,
    r"\text{Construct: } R_1 $\\$ $\\$  R_1 = \begin{bmatrix} R_{1, 1} \\ R_{1, 2}\end{bmatrix} = \begin{bmatrix} G_0 \phi_1 -\mu K_0 \phi_1 \\[1ex] \frac{1}{2}(1 - \phi_1^T K_0 \phi_1) \end{bmatrix}",
)
x.add_system(
    "ad1",
    SOLVER,
    r"\text{Solve: Adjoint System for } R_1 $\\$ $\\$  \text{w.r.t } \upsilon = [ \phi_1, \mu ] $\\$ $\\$  \begin{bmatrix} G_0 - \mu K_0 & - K_0 \phi_1 \\ - \phi_1^T K_0 & 0 \end{bmatrix} \begin{bmatrix} \psi_{\phi_1} \\ \psi_{\mu} \end{bmatrix} = - \begin{bmatrix} \dfrac{\partial a}{\partial \phi_1}^T \\[2ex] \dfrac{\partial a}{\partial \mu} \end{bmatrix}",
)
x.add_system(
    "R0",
    FUNC,
    r"\text{Construct: } R_0 $\\$ $\\$ R_0 = K_0 u_0 - f",
)
x.add_system(
    "ad0",
    SOLVER,
    r"\text{Solve: Adjoint System for } R_0 $\\$ $\\$ \text{w.r.t } \upsilon = u_0 $\\$ $\\$ K_0 \psi_{u_0} = - [\psi_{\phi_1}^T \dfrac{\partial G_0}{\partial u_0} \phi_1]^T",
)
x.add_system(
    "da",
    DOE,
    r"\text{Total Derivative for Koiter Factor $a$ :} $\\$ $\\$ \dfrac{d a}{d x_i} = \left(\dfrac{\partial a}{\partial x_i} + \psi_{\phi_1}^T \dfrac{\partial R_{1,1}}{\partial x_i} + \psi_{\mu} \dfrac{\partial R_{1,2}}{\partial x_i} + \psi_{u_0}^T \dfrac{\partial R_0}{\partial x_i}\right) \bigg\rvert_{\substack{\partial \upsilon_1 = 0\\ \partial u_0 = 0}}",
)

x.connect("R1", "ad1", r"\dfrac{\partial R_1}{\partial \upsilon}")
x.connect("R1", "ad0", r"R_1")
x.connect(
    "R1",
    "da",
    r"\dfrac{\partial R_{1,1}}{\partial x_i}, \dfrac{\partial R_{1,2}}{\partial x_i}",
)
x.connect("ad1", "da", r"\psi_{\phi_1}, \psi_{\mu}")
x.connect("ad1", "ad0", r"\psi_{\phi_1}, \psi_{\mu}")
x.connect("R0", "ad0", r"\dfrac{\partial R_0}{\partial u_0}")
x.connect("R0", "da", r"\dfrac{\partial R_0}{\partial x_i}")
x.connect("ad0", "da", r"\psi_{u_0}")


x.add_input("R1", r"K_0, G_0, \phi_1, \lambda_c")
x.add_input("ad1", r"K_0, K_1, G_0, \phi_1, \lambda_c")
x.add_input("R0", r"K_0, u_0, f")
x.add_input("ad0", r"K_1")
x.add_input(
    "da",
    r"\frac{\partial K_0}{\partial x_i} (u, v), \frac{\partial K_1}{\partial x_i} (u, v), \frac{\partial G_0}{\partial x_i} (u, v), \phi_1, \lambda_c",
)

# x.add_output("ad1", r"\psi_{\phi_1}, \psi_{\mu}", side=RIGHT)
# x.add_output("ad0", r"\psi_{u_0}", side=RIGHT)
x.add_output("da", r"\dfrac{d a}{d x_i}", side=RIGHT)

x.add_process(["R1", "ad1"], arrow=True)
x.add_process(["ad1", "ad0"], arrow=True)
# x.add_process(["ad1", "right_output_ad1"], arrow=True)
x.add_process(["ad1", "da"], arrow=True)
x.add_process(["R0", "ad0"], arrow=True)
x.add_process(["R0", "da"], arrow=True)
# x.add_process(["ad0", "right_output_ad0"], arrow=True)
x.add_process(["ad0", "da"], arrow=True)
x.add_process(["da", "right_output_da"], arrow=True)

x.write("xdsm", cleanup=False, quiet=True)
x.write_sys_specs("sink_specs")

images = convert_from_path("xdsm.pdf", dpi=600)
images[0].save("./output/koiter_a_xdsm.png")


##################################################
x = XDSM(
    auto_fade={
        # "inputs": "none",
        "outputs": "connected",
        "connections": "outgoing",
        # "processes": "none",
    }
)

x.add_system(
    "R2",
    FUNC,
    r"\text{Construct: } R_2 $\\$ $\\$ R_2 = \begin{bmatrix} R_{2, 1} \\ R_{2, 2}\end{bmatrix} = \begin{bmatrix} (G_0 - \mu K_0) u_2 + \nu K_0 u_1 - \mu (\tfrac{1}{2} K_1 + G_1) u_1 \\[1ex] u_1^T K_0 u_2\end{bmatrix}",
)
x.add_system(
    "ad2",
    SOLVER,
    r"\text{Solve: Adjoint System for } R_2 $\\$ $\\$ \text{w.r.t } \upsilon_2 = [ u_2, \nu ] $\\$ \begin{bmatrix} G_0 - \mu K_0 & K_0 u_1 \\ u_1^T K_0 & 0 \end{bmatrix} \begin{bmatrix} \psi_{u_2} \\ \psi_{\nu} \end{bmatrix} = - \begin{bmatrix} \dfrac{\partial b}{\partial u_2}^T \\[2ex] \dfrac{\partial b}{\partial \nu} \end{bmatrix}",
)
x.add_system(
    "R1",
    FUNC,
    r"\text{Construct: } R_1 $\\$ $\\$ R_1 = \begin{bmatrix} R_{1, 1} \\ R_{1, 2}\end{bmatrix} = \begin{bmatrix} G_0 \phi_1 -\mu K_0 \phi_1 \\[1ex] \frac{1}{2}(1 - \phi_1^T K_0 \phi_1) \end{bmatrix}",
)
x.add_system(
    "ad1",
    SOLVER,
    r"\text{Solve: Adjoint System for $R_1$ and $\hat{b}$} $\\$ $\\$ \text{ w.r.t } \upsilon_1 = [u_1, \mu] $\\$ $\\$  \text{where } \hat{b} = \left(b + \psi_{u_2}^T R_{2,1} + \psi_{\nu} R_{2,2}\right) $\\$  $\\$ \begin{bmatrix} \|\phi_1\| (G_0 - \mu K_0) & - \|\phi_1\| K_0 u_1 \\ - \|\phi_1\|^2 u_1^T K_0 & 0 \end{bmatrix} \begin{bmatrix} \psi_{u_1} \\ \psi_{\mu}\end{bmatrix} = - \begin{bmatrix} [\dfrac{\partial \hat{b}}{\partial u_1} \bigg\rvert_{\partial \upsilon_2 = 0}]^T \\[2ex] \dfrac{\partial \hat{b}}{\partial \mu} \bigg\rvert_{\partial \upsilon_2 = 0} \end{bmatrix}",
)
x.add_system(
    "db",
    DOE,
    r"\text{Total Derivative for Koiter Factor $b$ :} $\\$ $\\$ \dfrac{d b}{d x_i} = \left(\dfrac{\partial b}{\partial x_i} + \psi_{u_2}^T \dfrac{\partial R_{2,1}}{\partial x_i} + \psi_{\nu} \dfrac{\partial R_{2,2}}{\partial x_i} + \psi_{u_1}^T \dfrac{\partial R_{1,1}}{\partial x_i} + \psi_{\mu} \dfrac{\partial R_{1,2}}{\partial x_i}\right) \bigg\rvert_{\substack{\partial \upsilon_1 = 0\\ \partial \upsilon_2 = 0}}",
)

x.connect("R2", "ad2", r"\dfrac{\partial R_2}{\partial \upsilon_2}")
x.connect("R2", "ad1", r"R_2")
x.connect(
    "R2",
    "db",
    r"\dfrac{\partial R_{2,1}}{\partial x_i}, \dfrac{\partial R_{2,2}}{\partial x_i}",
)
x.connect("ad2", "db", r"\psi_{u_2}, \psi_{\nu}")
x.connect("ad2", "ad1", r"\psi_{u_2}, \psi_{\nu}")
x.connect("R1", "ad1", r"\dfrac{\partial R_1}{\partial \upsilon_1}")
x.connect(
    "R1",
    "db",
    r"\dfrac{\partial R_{1,1}}{\partial x_i}, \dfrac{\partial R_{1,2}}{\partial x_i}",
)
x.connect("ad1", "db", r"\psi_{u_1}, \psi_{\mu}")

x.add_input("R2", r"K_0, K_1, G_0, G_1, u_1, u_2, \lambda_c, \nu")
x.add_input("ad2", r"K_0, K_1, G_0, G_1, \phi_1, u_1, u_2, \lambda_c")
x.add_input("R1", r"K_0, G_0, \phi_1, \lambda_c")
x.add_input("ad1", r"b, K_0, K_1, K_{11}, G_0, G_1, \phi_1, u_1, u_2, \lambda_c, \nu")
x.add_input(
    "db",
    r"\frac{\partial K_0}{\partial x_i} (u, v), \frac{\partial K_1}{\partial x_i} (u, v), \frac{\partial K_{11}}{\partial x_i} (u, v), \frac{\partial G_0}{\partial x_i} (u, v),  \frac{\partial G_1}{\partial x_i} (u, v),\phi_1, u_1, u_2, \lambda_c, \nu",
)

# x.add_output("ad2", r"\psi_{u_2}, \psi_{\nu}", side=RIGHT)
# x.add_output("ad1", r"\psi_{u_1}, \psi_{\mu}", side=RIGHT)
x.add_output("db", r"\dfrac{d b}{d x_i}", side=RIGHT)

x.add_process(["R2", "ad2"], arrow=True)
x.add_process(["ad2", "ad1"], arrow=True)
# x.add_process(["ad2", "right_output_ad2"], arrow=True)
x.add_process(["ad2", "db"], arrow=True)
x.add_process(["R1", "ad1"], arrow=True)
x.add_process(["R1", "db"], arrow=True)
# x.add_process(["ad1", "right_output_ad1"], arrow=True)
x.add_process(["ad1", "db"], arrow=True)
x.add_process(["db", "right_output_db"], arrow=True)

x.write("xdsm", cleanup=False, quiet=True)
x.write_sys_specs("sink_specs")

images = convert_from_path("xdsm.pdf", dpi=600)
images[0].save("./output/koiter_b_xdsm.png")

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
x.add_system("Equilibrium", SOLVER, r"\text{Solve}: K_0 u_0 = f")
x.add_system("G0", FUNC, r"\text{Assembly} $\\$ G_0")
x.add_system(
    "Linear Buckling",
    SOLVER,
    r"\text{Solve: Linear Buckling System} $\\$ (K_0 + \lambda_c G_0) \phi_1 = 0",
)
x.add_system("K1", FUNC, r"\text{Assemble} $\\$ K_1, K_2, G_1")
x.add_system("Koiter-a", DOE, r"\text{Koiter Asymptotic} $\\$ \text{Theory}")
x.add_system(
    "Post Buckling",
    SOLVER,
    r"\text{Solve: Post-Buckling System} $\\$ \text{with} \ K_0\text{-orthonormal}",
)
x.add_system("Koiter-b", DOE, r"\text{Koiter Asymptotic} $\\$ \text{Theory}")

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
x.connect("K1", "Koiter-b", "K_1, K_2")


x.add_input("K0", r"x")
x.add_input("Equilibrium", r"f")
x.add_input("G0", r"x")
x.add_input("K1", r"x")

x.add_output("Linear Buckling", r"\lambda_c")
x.add_output("Linear Buckling", r"u_1", side=RIGHT)
x.add_output("Koiter-a", r"a")
x.add_output("Post Buckling", r"u_2", side=RIGHT)
x.add_output("Koiter-b", r"b")

x.add_process(["K0", "Equilibrium", "G0", "Linear Buckling"], arrow=True)
x.add_process(["Linear Buckling", "K1", "Koiter-a", "left_output_Koiter-a"], arrow=True)
x.add_process(["K1", "Post Buckling", "Koiter-b", "left_output_Koiter-b"], arrow=True)
x.add_process(["Linear Buckling", "left_output_Linear Buckling"], arrow=True)
x.add_process(["Linear Buckling", "right_output_Linear Buckling"], arrow=True)
x.add_process(["Post Buckling", "right_output_Post Buckling"], arrow=True)

x.write("xdsm", cleanup=False, quiet=True)
x.write_sys_specs("sink_specs")

images = convert_from_path("xdsm.pdf", dpi=500)
images[0].save("./output/xdsm.png")

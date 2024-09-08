import numpy as np


def shape_functions(xi, eta):
    N = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )
    Nxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
    Neta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])

    return N, Nxi, Neta


def populate_Be_and_He(nelems, xi, eta, xe, ye, Be, He):
    """
    Populate B matrices for all elements at a quadrature point
    """
    J = np.zeros((nelems, 2, 2))
    invJ = np.zeros(J.shape)

    N, Nxi, Neta = shape_functions(xi, eta)

    # Compute the Jacobian transformation at each quadrature points
    J[:, 0, 0] = np.dot(xe, Nxi)
    J[:, 1, 0] = np.dot(ye, Nxi)
    J[:, 0, 1] = np.dot(xe, Neta)
    J[:, 1, 1] = np.dot(ye, Neta)

    # Compute the inverse of the Jacobian
    detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    invJ[:, 0, 0] = J[:, 1, 1] / detJ
    invJ[:, 0, 1] = -J[:, 0, 1] / detJ
    invJ[:, 1, 0] = -J[:, 1, 0] / detJ
    invJ[:, 1, 1] = J[:, 0, 0] / detJ

    # Compute the derivative of the shape functions w.r.t. xi and eta
    # [Nx, Ny] = [Nxi, Neta]*invJ
    Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
    Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)

    # Set the B matrix for each element
    Be[:, 0, ::2] = Nx
    Be[:, 1, 1::2] = Ny
    Be[:, 2, ::2] = Ny
    Be[:, 2, 1::2] = Nx

    He[:, 0, ::2] = N
    He[:, 1, 1::2] = N

    return detJ


def populate_Be_and_Te(nelems, xi, eta, xe, ye, Be, Te):
    """
    Populate B matrices for all elements at a quadrature point
    """
    J = np.zeros((nelems, 2, 2))
    invJ = np.zeros(J.shape)

    Nxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
    Neta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])

    # Compute the Jacobian transformation at each quadrature points
    J[:, 0, 0] = np.dot(xe, Nxi)
    J[:, 1, 0] = np.dot(ye, Nxi)
    J[:, 0, 1] = np.dot(xe, Neta)
    J[:, 1, 1] = np.dot(ye, Neta)

    # Compute the inverse of the Jacobian
    detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    invJ[:, 0, 0] = J[:, 1, 1] / detJ
    invJ[:, 0, 1] = -J[:, 0, 1] / detJ
    invJ[:, 1, 0] = -J[:, 1, 0] / detJ
    invJ[:, 1, 1] = J[:, 0, 0] / detJ

    # Compute the derivative of the shape functions w.r.t. xi and eta
    # [Nx, Ny] = [Nxi, Neta]*invJ
    Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
    Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)

    # Set the B matrix for each element
    Be[:, 0, ::2] = Nx
    Be[:, 1, 1::2] = Ny
    Be[:, 2, ::2] = Ny
    Be[:, 2, 1::2] = Nx

    # Set the entries for the stress stiffening matrix
    for i in range(nelems):
        Te[i, 0, :, :] = np.outer(Nx[i, :], Nx[i, :])
        Te[i, 1, :, :] = np.outer(Ny[i, :], Ny[i, :])
        Te[i, 2, :, :] = np.outer(Nx[i, :], Ny[i, :]) + np.outer(Ny[i, :], Nx[i, :])

    return detJ


def populate_nonlinear_strain_and_Be(Be, ueve, Be_nl, strain_nl=None, strain_li=None):
    """
    Populate B matrices for all elements at a quadrature point
    """
    Nx = Be[:, 0, ::2]
    Ny = Be[:, 1, 1::2]

    ue = ueve[:, ::2]
    ve = ueve[:, 1::2]

    ux = np.sum(ue * Nx, axis=1)[..., np.newaxis]
    uy = np.sum(ue * Ny, axis=1)[..., np.newaxis]

    vx = np.sum(ve * Nx, axis=1)[..., np.newaxis]
    vy = np.sum(ve * Ny, axis=1)[..., np.newaxis]

    if strain_li is not None:
        strain_li[:, 0] = (ux).flatten()
        strain_li[:, 1] = (vy).flatten()
        strain_li[:, 2] = (uy + vx).flatten()

    if strain_nl is not None:
        strain_nl[:, 0] = (ux + 0.5 * (ux**2 + vx**2)).flatten()
        strain_nl[:, 1] = (vy + 0.5 * (uy**2 + vy**2)).flatten()
        strain_nl[:, 2] = (uy + vx + ux * uy + vx * vy).flatten()

    # Set the B matrix for each element
    Be_nl[:, 0, ::2] = ux * Nx
    Be_nl[:, 0, 1::2] = vx * Nx

    Be_nl[:, 1, ::2] = uy * Ny
    Be_nl[:, 1, 1::2] = vy * Ny

    Be_nl[:, 2, ::2] = uy * Nx + ux * Ny
    Be_nl[:, 2, 1::2] = vy * Nx + vx * Ny

    return


def compute_detJ(nelems, xi, eta, xe, ye):
    J = np.zeros((nelems, 2, 2))

    N, Nxi, Neta = shape_functions(xi, eta)

    # Compute the Jacobian transformation at each quadrature points
    J[:, 0, 0] = np.dot(xe, Nxi)
    J[:, 1, 0] = np.dot(ye, Nxi)
    J[:, 0, 1] = np.dot(xe, Neta)
    J[:, 1, 1] = np.dot(ye, Neta)

    # Compute the inverse of the Jacobian
    detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]

    return detJ


def interp_from_element(xi, eta, xe):
    N, Nxi, Neta = shape_functions(xi, eta)

    return np.dot(xe, N)


def populate_thermal_Be_and_He(nelems, xi, eta, xe, ye, Be, He):
    """
    Populate B matrices for all elements at a quadrature point
    """
    J = np.zeros((nelems, 2, 2))
    invJ = np.zeros(J.shape)

    N, Nxi, Neta = shape_functions(xi, eta)

    # Compute the Jacobian transformation at each quadrature points
    J[:, 0, 0] = np.dot(xe, Nxi)
    J[:, 1, 0] = np.dot(ye, Nxi)
    J[:, 0, 1] = np.dot(xe, Neta)
    J[:, 1, 1] = np.dot(ye, Neta)

    # Compute the inverse of the Jacobian
    detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    invJ[:, 0, 0] = J[:, 1, 1] / detJ
    invJ[:, 0, 1] = -J[:, 0, 1] / detJ
    invJ[:, 1, 0] = -J[:, 1, 0] / detJ
    invJ[:, 1, 1] = J[:, 0, 0] / detJ

    # Compute the derivative of the shape functions w.r.t. xi and eta
    # [Nx, Ny] = [Nxi, Neta]*invJ
    Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
    Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)

    # Set the B matrix for each element
    Be[:, 0, :] = Nx
    Be[:, 1, :] = Ny
    He[:, :] = N

    return detJ

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
        mu, u1 = sparse.linalg.eigsh(Gr, M=Kr, k=1, which="SM", sigma=sigma)
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

    M = sparse.csc_matrix((Me.flatten(), (self.i, self.j)))

    if self.M00 is not None:
        M += self.M00

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


def get_dG0dx_dK0dx_du0dx_cs(self, x, K, G, u0, Q0, dh=1e-30):
    """
    Compute the derivative of the stiffness matrix times the vectors u
    """
    du0dx = np.zeros((2 * self.nnodes, self.nnodes))
    dK0dx = np.zeros((2 * self.nnodes, self.nnodes))
    dG0dx = np.zeros((2 * self.nnodes, self.nnodes))

    t0 = time.time()
    for n in range(self.nnodes):
        x_cs = x.copy().astype(complex)
        x_cs[n] += dh * 1j

        rho = self.fltr.apply(x_cs)

        K1 = self.get_stiffness_matrix(rho)
        K1r = self.reduce_matrix(K1)

        u1r = sparse.linalg.spsolve(K1r, self.fr)

        u01 = self.full_vector(u1r)
        G1 = self.get_stress_stiffness_matrix(rho, u01)

        du0dx[:, n] = (u01 - u0).imag / dh
        dG0dx[:, n] = (G1 - G).imag @ Q0 / dh
        dK0dx[:, n] = (K1 - K).imag @ Q0 / dh

        _print_progress(n, self.nnodes, t0, "dQ0/dx")

    return dK0dx, dG0dx, du0dx


@timeit(print_time=True)
def get_dG0dx_dK0dx_du0dx_cs(self, x, K, G, u0, Q0, dh=1e-30, du0=False):
    """
    Compute the derivative of the stiffness matrix times the vectors u
    """
    # Preallocate memory for results before parallelization
    ndv = x.size
    dG0dx = np.zeros((2 * self.nnodes, ndv), dtype=x.dtype)
    dK0dx = np.zeros((2 * self.nnodes, ndv), dtype=x.dtype)
    du0dx = np.zeros((2 * self.nnodes, ndv), dtype=x.dtype) if du0 else None

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
        du0dx_n = (u01 - u0).imag / dh if du0 else 0

        _print_progress(ii, self.nnodes, t0, "dQ0/dx")

        return dG0dx_n, dK0dx_n, du0dx_n

    t0 = time.time()

    # Parallelize the loop
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(iterator)(ii) for ii in range(ndv)
    )

    # Unpack results efficiently
    for n, (dG0dx_n, dK0dx_n, du0dx_n) in enumerate(results):
        dG0dx[:, n] = dG0dx_n
        dK0dx[:, n] = dK0dx_n
        if du0:
            du0dx[:, n] = du0dx_n

    return dK0dx, dG0dx, du0dx


def get_dadx(self, rhoE, Q0, K1):
    G = self.get_stress_stiffness_matrix(rhoE, Q0)
    u1_norm = np.linalg.norm(Q0)

    # dK1dQ0_cd = self.get_dK1dQ0_cd(rhoE, Q0, Q0, Q0)
    # dK1dQ0_cs = self.get_dK1dQ0_cs(rhoE, K1, Q0, Q0, Q0)
    dK1dQ0 = G @ Q0 / u1_norm

    self.initialize_adjoint()
    Qb = Q0.T @ (K1 + K1.T) + dK1dQ0
    self.Qrb[:, 0] = self.reduce_vector(Qb)
    self.finalize_adjoint()

    dK1dx = self.get_dK1dx(rhoE, Q0, Q0, Q0)
    dadx = 1.5 * (self.xb + dK1dx)

    return dadx
  
  
def get_dadx(self, x, rho, rhoE, l0, Q0, K1, G1):
    t0 = time.time()
    Ar = self.Gr + self.Kr / l0
    Q0r = self.reduce_vector(Q0)
    Lr = (self.Kr @ Q0r).reshape(-1, 1)
    K1r = self.reduce_matrix(K1)
    mat = sparse.bmat([[Ar, -Lr], [-Lr.T, [0]]], format="csc")

    # dK1dQ0_cd = self.get_dK1dQ0_cd(rhoE, Q0, Q0, Q0)
    # dK1dQ0_cd = self.reduce_vector(dK1dQ0_cd)

    # dK1dQ0_cs = self.get_dK1dQ0_cs(rhoE, K1, Q0, Q0, Q0)
    # dK1dQ0_cs = self.reduce_vector(dK1dQ0_cs)

    G1r = self.reduce_matrix(G1)
    dK1dQ0 = G1r @ Q0r

    Qb = (K1r.T + K1r) @ Q0r + dK1dQ0

    rhs = np.hstack([Qb, 0])
    psi = sparse.linalg.spsolve(mat, -1.5 * rhs)

    psi0 = self.full_vector(psi[:-1])
    psi1 = psi[-1]

    dGdx1 = self.get_stress_stiffness_matrix_xuderiv(rhoE, self.u, psi0, Q0)
    dKdx1 = self.get_stiffness_matrix_deriv(rhoE, psi0, Q0)
    dKdx2 = self.get_stiffness_matrix_deriv(rhoE, Q0, Q0)
    dK1dx = self.get_dK1dx(rhoE, Q0, Q0, Q0)

    dGdx1 = self.fltr.apply_gradient(dGdx1, x)
    dKdx1 = self.fltr.apply_gradient(dKdx1, x)
    dKdx2 = self.fltr.apply_gradient(dKdx2, x)

    dadx = (dGdx1 + dKdx1 / l0) - 0.5 * psi1 * dKdx2 + 1.5 * dK1dx
    t1 = time.time()
    ic(t1 - t0)
    
    t2 = time.time()
    self.initialize_adjoint()
    K1r = self.reduce_matrix(K1)
    G1r = self.reduce_matrix(G1)
    dK1dQ0 = G1r @ Q0r
    Qb = (K1r.T + K1r) @ Q0r + dK1dQ0
    self.Qrb[:, 0] = Qb
    self.finalize_adjoint()
    dadx = 1.5 *(self.xb + dK1dx)
    t3 = time.time()
    ic(t3 - t2)

    return dadx
  
  
  
  # check if Ar is singular
# Ar = K0r + lam_c * G0r
if np.linalg.matrix_rank(Ar.toarray()) < Ar.shape[0]:
    print("Ar is singular")
else:
    print("Ar is not singular")
exit()

    # for i in range(4):
    #     detJ = self.detJ[:, i]
    #     Be = self.Be[:, :, :, i]
    #     Be1 = np.zeros((self.nelems, 3, 8))
    #     populate_nonlinear_strain_and_Be(Be, ue1, Be1)

    #     # strain_0 = np.einsum("nij,nj -> ni", Be, ue0)
    #     strain_1 = np.einsum("nij,nj -> ni", Be, ue1)
    #     strain_11 = np.einsum("nij,nj -> ni", Be1, ue1)
    #     strain_12 = np.einsum("nij,nj -> ni", Be1, ue2)
    #     strain_02 = np.einsum("nij,nj -> ni", Be, ue2)

    #     # stress_0 = np.einsum("nij,nj -> ni", CK, strain_0)
    #     stress_1 = np.einsum("nij,nj -> ni", CK, strain_1)
    #     stress_2 = np.einsum("nij,nj -> ni", CK, (strain_02 + 0.5 * strain_11))

    #     an += np.einsum("ni,ni -> n", stress_1, strain_11)
    #     b1 = np.einsum("ni,ni -> n", stress_2, strain_11)
    #     b2 = np.einsum("ni,ni -> n", stress_1, strain_12)
    #     bn += b1 + 2 * b2
    #     d2 += np.einsum("ni,ni -> n", stress_1, strain_1)

    # an = np.sum(an)
    # bn = np.sum(bn)
    # # d1 = np.sum(d1)
    # d2 = np.sum(d2)

    # # a11 = 1.5 * an / d1
    # a22 = 1.5 * an / d2

    # # b11 = bn / d1
    # b22 = bn / d2
    # t1 = time.time()
    # ic(t1 - t0)

    # ic(a11, a22)
    # ic(b11, b22)

    # indy = 2 * np.nonzero(self.f[1::2])[0] + 1
    # indy = indy[len(indy) // 2]
    # indx = indy - 1

    # xi = np.linspace(-1e1, 1e1, 100)

    # lam = (1 + a * xi + b * xi**2) * lam_c

    # ux = u0[indx] * lam + u1[indx] * xi + u2[indx] * xi**2
    # uy = u0[indy] * lam + u1[indy] * xi + u2[indy] * xi**2

    # lam_0 = np.linspace(0, lam_c, 100)
    # u0x = u0[indx] * lam_0
    # u0y = u0[indy] * lam_0

    # fig, ax = plt.subplots(1, 4, figsize=(16, 3), tight_layout=True)
    # ax[0].plot(xi, lam / lam_c, color="k")
    # ax[0].plot([0, 0], [0, 1], color="b")
    # ax[0].scatter(0, 1, color="r", zorder=10)
    # ax[0].set_xlabel(r"$\xi$")
    # ax[0].set_ylabel(r"$\lambda/\lambda_c$")
    # ax[0].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # ax[1].plot(u0x, lam_0, color="b")
    # ax[1].plot(ux, lam, color="k")
    # ax[1].scatter(u0[indx] * lam_c, lam_c, color="r", zorder=10)
    # ax[1].set_xlabel(r"$u_x$")
    # ax[1].set_ylabel(r"$\lambda$")
    # ax[1].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # ax[2].plot(u0y, lam_0, color="b")
    # ax[2].plot(uy, lam, color="k")
    # ax[2].scatter(u0[indy] * lam_c, lam_c, color="r", zorder=10)
    # ax[2].set_xlabel(r"$u_y$")
    # ax[2].set_ylabel(r"$\lambda$")
    # ax[2].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    # ax[2].invert_xaxis()

    # res_norm0 = np.zeros(len(xi))
    # res_norm = np.zeros(len(xi))
    # # for i in range(len(xi)):
    # #     ui = u0 * lam[i] + u1 * xi[i] + u2 * xi[i] ** 2
    # #     u0i = u0 * lam[i]
    # #     res = self.getResidual(self.rhoE, ui, lam[i])[1]
    # #     res_norm[i] = np.linalg.norm(res)

    # #     res = self.getResidual(self.rhoE, u0i, lam_c)[1]
    # #     res_norm0[i] = np.linalg.norm(res)
    # #     print(f"Residual[{i:3d}]  {res_norm[i]:15.5e}")

    # ax[3].semilogy(xi, res_norm, color="b")
    # ax[3].semilogy(xi, res_norm0, color="k")
    # ax[3].set_xlabel(r"$\xi$")
    # ax[3].set_ylabel(r"$||R||$")
    # ax[3].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # plt.savefig("load-deflection.pdf", bbox_inches="tight")

    # def plot_u():
    #     xi = 1e1
    #     lam = (1 + a * xi + b * xi**2) * lam_c
    #     u = u0 * lam + u1 * xi + u2 * xi**2
    #     u_lin = u0 * lam
    #     u_list = [u0, u1, u0 * lam_c, u_lin, u]

    #     fig, ax = plt.subplots(1, 5, figsize=(15, 4))
    #     levels = np.linspace(0.0, 1.0, 26)
    #     title = [
    #         r"$u_0$",
    #         r"$u_1$",
    #         r"$u_0 \lambda_c$",
    #         r"$u_0 \lambda$",
    #         r"$u$ at $\lambda = " + f"{lam:.2f}$",
    #     ]
    #     for i, u in enumerate(u_list):
    #         self.plot(
    #             self.rho, ax=ax[i], u=u, levels=levels, extend="max", cmap="Greys"
    #         )
    #         ax[i].set_title(title[i])
    #         ax[i].set_xticks([])
    #         ax[i].set_yticks([])

    #     plt.savefig("u.pdf", bbox_inches="tight")

    # plot_u()



        nu = mu * u1r @ (0.5 * K1r + G1r) @ u1r * Q0_norm**2
        # # check if nu equals to self.nu
        ic(np.allclose(nu, self.nu))
        ic(nu, self.nu)

        # construct the block matrix A in sparse format
        a00 = - mu * (K1r + 2 * G1r) + nu * K0r + mu * Q0_norm**2 * np.outer(K0r @ u1r, (0.5*K1r + 0.5*K1r.T + 2 * G1r) @ u1r)
        a01 = G0r - mu * K0r
        a02 = (-K0r @ u2r - (0.5 * K1r + G1r) @ u1r + nu / mu * K0r @ u1r).reshape(-1, 1)
        a03 = (K0r @ u1r).reshape(-1, 1)

        a10 = Q0_norm * (G0r - mu * K0r)
        a11 = np.zeros((lr, lr))
        a12 = (-K0r @ Q0r).reshape(-1, 1)
        a13 = np.zeros((lr, 1))

        a20 = u2r.T @ K0r
        a21 = u1r.T @ K0r

        a30 = -(Q0_norm**2) * u1r.T @ K0r
        a31 = np.zeros((1, lr))

        A = sparse.bmat(
            [
                [a00, a01, a02],
                [a10, a11, a12],
                # [a20, a21, [0], [0]],
                [a30, a31, [0]],
            ],
            format="csc",
        )

        pbpu1 = 2 * Q0_norm**2 * (u2r @ (K1r + K1r.T + G1r) + u1r @ K11r)

        # pbpu1_cs = self.get_dbdu1_cs(rhoE, l0, Q0)
        # pbpu1_cd = self.get_dbdu1_cd(rhoE, l0, Q0)
        # pbpu1_cs = self.reduce_vector(pbpu1_cs)
        # pbpu1_cd = self.reduce_vector(pbpu1_cd)

        # ic(np.allclose(pbpu1, pbpu1_cs))
        # ic(np.allclose(pbpu1, pbpu1_cd))
        # ic(np.allclose(pbpu1_cs, pbpu1_cd))
        # for i in range(10):
        #     print(pbpu1[i])
        #     print(pbpu1_cs[i])
        #     print(pbpu1_cd[i])
        #     print()

        # exit()

        pbpu2 = Q0_norm**2 * u1r @ (2 * K1r + K1r.T)

        # pbpu2_cs = self.get_dbdu2_cs(rhoE, l0, Q0)
        # pbpu2_cs = self.reduce_vector(pbpu2_cs)
        # ic(np.allclose(pbpu2, pbpu2_cs))

        # for i in range(10):
        #     print(pbpu2[i])
        #     print(pbpu2_cs[i])
        #     print()

        rhs = np.hstack([pbpu1, pbpu2, 0])

        psi = sparse.linalg.spsolve(A.T, -rhs)
        pu1 = self.full_vector(psi[:lr])
        pu2 = self.full_vector(psi[lr : 2 * lr])
        pmu = psi[-1]
        # pnu = psi[-1]

        pu1_dG0_u2 = self.get_stress_stiffness_matrix_xuderiv(rhoE, u0, pu1, u2)
        pu1_dK0_u2 = self.get_stiffness_matrix_deriv(rhoE, pu1, u2)
        pu1_dK0_u1 = self.get_stiffness_matrix_deriv(rhoE, pu1, u1)

        pu1_dK1_u1 = self.get_dK1dx(rhoE, u1, pu1, u1)
        pu1_dG1_u1 = self.get_stress_stiffness_matrix_xderiv(rhoE, u1, pu1, u1)

        # pu1_dG1_u1_cs = self.get_dG1dx_cs(x, rhoE, u1, pu1, u1)
        # pu1_dG1_u1_cd = self.get_dG1dx_cd(x, rhoE, u1, pu1, u1)
        # ic(np.allclose(pu1_dG1_u1, pu1_dG1_u1_cs))
        # ic(np.allclose(pu1_dG1_u1, pu1_dG1_u1_cd))
        # ic(np.allclose(pu1_dG1_u1_cs, pu1_dG1_u1_cd))

        # for i in range(10):
        #     print(pu1_dG1_u1[i])
        #     print(pu1_dG1_u1_cs[i])
        #     print(pu1_dG1_u1_cd[i])
        #     print()

        # exit()

        # p0 = sparse.linalg.spsolve(K0r, np.zeros(lr))
        # p0 = self.full_vector(p0)
        # p0_dK0_u0 = self.get_stiffness_matrix_deriv(rhoE, p0, u0)
        # p0_dK0_u0 = self.fltr.apply_gradient(p0_dK0_u0, x)
        # # check if p0 equals to zero
        # ic(np.allclose(p0, 0))

        pu2_dG0_Q0 = self.get_stress_stiffness_matrix_xuderiv(rhoE, u0, pu2, Q0)
        pu2_dK0_Q0 = self.get_stiffness_matrix_deriv(rhoE, pu2, Q0)

        u1_dK0_u2 = self.get_stiffness_matrix_deriv(rhoE, u1, u2)
        Q0_dK0_Q0 = self.get_stiffness_matrix_deriv(rhoE, Q0, Q0)

        list1 = [pu1_dG0_u2, pu1_dK0_u2, pu1_dK0_u1, pu1_dG1_u1]
        lsit2 = [pu2_dG0_Q0, pu2_dK0_Q0, u1_dK0_u2, Q0_dK0_Q0]
        for dX in list1 + lsit2:
            dX = self.fltr.apply_gradient(dX, x)
            
        u1_dK1_u1 = self.get_dK1dx(rhoE, Q0, u1, u1)
        u1_dG1_u1 = self.get_stress_stiffness_matrix_xderiv(rhoE, u1, u1, u1)
        u1_dG1_u1 = self.fltr.apply_gradient(u1_dG1_u1, x)

        dbdx1 = (
            pu1_dG0_u2
            - mu * pu1_dK0_u2
            + nu * pu1_dK0_u1
            - mu * (0.5 * pu1_dK1_u1 + pu1_dG1_u1)
        )
        dbdx2 = (
            pu2_dG0_Q0 - mu * pu2_dK0_Q0 - 0.5 * pmu * Q0_dK0_Q0
        )  # + pmu * u1_dK0_u2 - 0.5 * pnu * Q0_dK0_Q0
        dbdx = pbpx + dbdx1 + dbdx2 + pu1 @ K0 @ u1 * mu * Q0_norm**2 * (0.5 * u1_dK1_u1 + u1_dG1_u1)


  # construct the block matrix A in sparse format
        a00 = - mu * (K1r + 2 * G1r) + nu * K0r 
        a01 = G0r - mu * K0r
        a02 = (-K0r @ u2r - (0.5 * K1r + G1r) @ u1r).reshape(-1, 1)
        a03 = (K0r @ u1r).reshape(-1, 1)

        a10 = Q0_norm * (G0r - mu * K0r)
        a11 = np.zeros((lr, lr))
        a12 = (-K0r @ Q0r).reshape(-1, 1)
        a13 = np.zeros((lr, 1))

        a20 = u2r.T @ K0r
        a21 = u1r.T @ K0r

        a30 = -(Q0_norm**2) * u1r.T @ K0r
        a31 = np.zeros((1, lr))

        A = sparse.bmat(
            [
                [a00, a01, a02, a03],
                [a10, a11, a12, a13],
                [a20, a21, [0], [0]],
                [a30, a31, [0], [0]],
            ],
            format="csc",
        )

        pbpu1 = 2 * Q0_norm**2 * (u2r @ (K1r + K1r.T + G1r) + u1r @ K11r)

        # pbpu1_cs = self.get_dbdu1_cs(rhoE, l0, Q0)
        # pbpu1_cd = self.get_dbdu1_cd(rhoE, l0, Q0)
        # pbpu1_cs = self.reduce_vector(pbpu1_cs)
        # pbpu1_cd = self.reduce_vector(pbpu1_cd)

        # ic(np.allclose(pbpu1, pbpu1_cs))
        # ic(np.allclose(pbpu1, pbpu1_cd))
        # ic(np.allclose(pbpu1_cs, pbpu1_cd))
        # for i in range(10):
        #     print(pbpu1[i])
        #     print(pbpu1_cs[i])
        #     print(pbpu1_cd[i])
        #     print()

        # exit()

        pbpu2 = Q0_norm**2 * u1r @ (2 * K1r + K1r.T)

        # pbpu2_cs = self.get_dbdu2_cs(rhoE, l0, Q0)
        # pbpu2_cs = self.reduce_vector(pbpu2_cs)
        # ic(np.allclose(pbpu2, pbpu2_cs))

        # for i in range(10):
        #     print(pbpu2[i])
        #     print(pbpu2_cs[i])
        #     print()

        rhs = np.hstack([pbpu1, pbpu2, 0, 0])

        psi = sparse.linalg.spsolve(A.T, -rhs)
        pu1 = self.full_vector(psi[:lr])
        pu2 = self.full_vector(psi[lr : 2 * lr])
        pmu = psi[-2]
        pnu = psi[-1]

        pu1_dG0_u2 = self.get_stress_stiffness_matrix_xuderiv(rhoE, u0, pu1, u2)
        pu1_dK0_u2 = self.get_stiffness_matrix_deriv(rhoE, pu1, u2)
        pu1_dK0_u1 = self.get_stiffness_matrix_deriv(rhoE, pu1, u1)

        pu1_dK1_u1 = self.get_dK1dx(rhoE, u1, pu1, u1)
        pu1_dG1_u1 = self.get_stress_stiffness_matrix_xderiv(rhoE, u1, pu1, u1)

        # pu1_dG1_u1_cs = self.get_dG1dx_cs(x, rhoE, u1, pu1, u1)
        # pu1_dG1_u1_cd = self.get_dG1dx_cd(x, rhoE, u1, pu1, u1)
        # ic(np.allclose(pu1_dG1_u1, pu1_dG1_u1_cs))
        # ic(np.allclose(pu1_dG1_u1, pu1_dG1_u1_cd))
        # ic(np.allclose(pu1_dG1_u1_cs, pu1_dG1_u1_cd))

        # for i in range(10):
        #     print(pu1_dG1_u1[i])
        #     print(pu1_dG1_u1_cs[i])
        #     print(pu1_dG1_u1_cd[i])
        #     print()

        # exit()

        # p0 = sparse.linalg.spsolve(K0r, np.zeros(lr))
        # p0 = self.full_vector(p0)
        # p0_dK0_u0 = self.get_stiffness_matrix_deriv(rhoE, p0, u0)
        # p0_dK0_u0 = self.fltr.apply_gradient(p0_dK0_u0, x)
        # # check if p0 equals to zero
        # ic(np.allclose(p0, 0))

        pu2_dG0_Q0 = self.get_stress_stiffness_matrix_xuderiv(rhoE, u0, pu2, Q0)
        pu2_dK0_Q0 = self.get_stiffness_matrix_deriv(rhoE, pu2, Q0)

        u1_dK0_u2 = self.get_stiffness_matrix_deriv(rhoE, u1, u2)
        Q0_dK0_Q0 = self.get_stiffness_matrix_deriv(rhoE, Q0, Q0)

        list1 = [pu1_dG0_u2, pu1_dK0_u2, pu1_dK0_u1, pu1_dG1_u1]
        lsit2 = [pu2_dG0_Q0, pu2_dK0_Q0, u1_dK0_u2, Q0_dK0_Q0]
        for dX in list1 + lsit2:
            dX = self.fltr.apply_gradient(dX, x)

        dbdx1 = (
            pu1_dG0_u2
            - mu * pu1_dK0_u2
            + nu * pu1_dK0_u1
            - mu * (0.5 * pu1_dK1_u1 + pu1_dG1_u1)
        )
        dbdx2 = pu2_dG0_Q0 - mu * pu2_dK0_Q0 + pmu * u1_dK0_u2 - 0.5 * pnu * Q0_dK0_Q0
        dbdx = pbpx + dbdx1 + dbdx2
        
        Ar = self.Gr + self.Kr / l0
        Q0r = self.reduce_vector(Q0)
        Lr = (self.Kr @ Q0r).reshape(-1, 1)
        K1r = self.reduce_matrix(self.K1)
        mat = sparse.bmat([[Ar, -Lr], [-Lr.T, [0]]], format="csc")

        # dK1dQ0_cd = self.get_dK1dQ0_cd(rhoE, Q0, Q0, Q0)
        # dK1dQ0_cd = self.reduce_vector(dK1dQ0_cd)

        # dK1dQ0_cs = self.get_dK1dQ0_cs(rhoE, K1, Q0, Q0, Q0)
        # dK1dQ0_cs = self.reduce_vector(dK1dQ0_cs)

        Qb = Q0r @ (K1r.T + 2 * K1r)
        rhs = np.hstack([Qb, 0])
        psi = sparse.linalg.spsolve(mat, -1.5 * rhs)

        psi0 = self.full_vector(psi[:-1])
        psi1 = psi[-1]

        dGdx1 = self.get_stress_stiffness_matrix_xuderiv(rhoE, self.u, psi0, Q0)
        dKdx1 = self.get_stiffness_matrix_deriv(rhoE, psi0, Q0)
        dKdx2 = self.get_stiffness_matrix_deriv(rhoE, Q0, Q0)
        dK1dx = self.get_dK1dx(rhoE, Q0, Q0, Q0)

        dGdx1 = self.fltr.apply_gradient(dGdx1, x)
        dKdx1 = self.fltr.apply_gradient(dKdx1, x)
        dKdx2 = self.fltr.apply_gradient(dKdx2, x)

        dadx = (dGdx1 + dKdx1 / l0) - 0.5 * psi1 * dKdx2 + 1.5 * dK1dx

        return dadx

    def get_dbdx(self, x, rhoE, l0, Q0, dl0dx, du0dx, dQ0dx, dh=1e-30):
        
        u0 = self.u
        u1 = self.u1
        u2 = self.u2
        u1r = self.reduce_vector(u1)
        u2r = self.reduce_vector(u2)
        Q0r = self.reduce_vector(Q0)
        
        K0 = self.K0
        K1 = self.K1
        K0r = self.reduce_matrix(self.K0)
        G0r = self.reduce_matrix(self.G0)
        K1r = self.reduce_matrix(self.K1)
        G1r = self.reduce_matrix(self.G1)
        K11r = self.reduce_matrix(self.K11)
        G2 = self.get_stress_stiffness_matrix(self.rho, u2)
        G2r = self.reduce_matrix(G2)

        lr = K0r.shape[0]
        mu = -1.0 / l0
        Q0_norm = np.linalg.norm(Q0)
        nu = mu * u1r @ (0.5 * K1r + G1r) @ u1r * Q0_norm**2

        Ar = G0r - mu * K0r
        Lr = (K0r @ u1r).reshape(-1, 1)
        mat = sparse.bmat([[Ar, Lr], [Lr.T, [0]]], format="csc")
        pbpu2 = Q0_norm**2 * u1r @ (2 * K1r + K1r.T)
        rhs = np.hstack([pbpu2, 0])

        psi = sparse.linalg.spsolve(mat.T, -rhs)
        pu2r = psi[:-1]
        pu2 = self.full_vector(pu2r)
        pnu = psi[-1]

        mat = sparse.bmat([[Ar*Q0_norm, -Lr*Q0_norm], [-Q0_norm**2*Lr.T, [0]]], format="csc")
        pbpu1 = 2 * Q0_norm**2 * (u2r @ (K1r + K1r.T + G1r) + u1r @ K11r)
        rhs = np.hstack([pbpu1, 0])

        psi = sparse.linalg.spsolve(mat.T, -rhs)
        pu1r = psi[:-1]
        pu1 = self.full_vector(pu1r)
        pmu = psi[-1]

        R1 = - mu * G1r @ u1r

        pu2_pR1pu1_cs = np.zeros(lr)

        for i in range(lr):
            u1r_1 = u1r.copy().astype(complex)
            u1r_1[i] += 1j * dh

            u1_1 = self.full_vector(u1r_1)

            rho = self.fltr.apply(x)

            G1_1 = self.get_stress_stiffness_matrix(rho, u1_1)     
            K1_1 = self.get_K1(rhoE, u1_1)

            G1r_1 = self.reduce_matrix(G1_1)
            K1r_1 = self.reduce_matrix(K1_1) 

            R1_1 =  - mu * G1r_1 @ u1r

            pu2_pR1pu1_cs[i] = pu2r @ (R1_1 - R1).imag / dh

        rhs1 =  - mu * pu2r @ (K1r.T)
        # check if rhs1 equals to pu2_pR1pu1_cs
        ic(np.allclose(rhs1, pu2_pR1pu1_cs))
        # exit()

        mat = sparse.bmat([[Ar*Q0_norm, -Lr*Q0_norm], [-Q0_norm**2*Lr.T, [0]]], format="csc")
        rhs1 = nu * pu2r @ K0r - mu * pu2r @ (K1r + K1r.T + G1r)
        rhs2 = - pu2r @ K0r @ u2r - pu2r @ (0.5 * K1r + G1r) @ u1r
        rhs = np.hstack([rhs1, rhs2])

        psi = sparse.linalg.spsolve(mat.T, -rhs)
        pu12r = psi[:-1]
        pu12 = self.full_vector(pu12r)
        pmu2 = psi[-1]

        pu2_dG0dx_u2 = self.get_stress_stiffness_matrix_xuderiv(rhoE, u0, pu2, u2)
        pu2_dG0dx_u2 = self.fltr.apply_gradient(pu2_dG0dx_u2, x)

        pu2_dK0dx_u2 = self.get_stiffness_matrix_deriv(rhoE, pu2, u2)
        pu2_dK0dx_u2 = self.fltr.apply_gradient(pu2_dK0dx_u2, x)

        pu2_dK0dx_u1 = self.get_stiffness_matrix_deriv(rhoE, pu2, u1)
        pu2_dK0dx_u1 = self.fltr.apply_gradient(pu2_dK0dx_u1, x)

        pu2_dK1_u1 = self.get_dK1dx(rhoE, Q0, pu2, u1)
        pu2_dG1_u1 = self.get_stress_stiffness_matrix_xderiv(rhoE, u1, pu2, u1)
        pu2_dG1_u1 = self.fltr.apply_gradient(pu2_dG1_u1, x)

        pu2_pR1px = pu2_dG0dx_u2 - mu * pu2_dK0dx_u2 + nu * pu2_dK0dx_u1 - mu * (0.5*pu2_dK1_u1 + pu2_dG1_u1)

        pu12_dG0dx_Q0 = self.get_stress_stiffness_matrix_xuderiv(rhoE, u0, pu12, Q0)
        pu12_dG0dx_Q0 = self.fltr.apply_gradient(pu12_dG0dx_Q0, x)

        pu12_dK0dx_Q0 = self.get_stiffness_matrix_deriv(rhoE, pu12, Q0)
        pu12_dK0dx_Q0 = self.fltr.apply_gradient(pu12_dK0dx_Q0, x)

        Q0_dK0dx_Q0 = self.get_stiffness_matrix_deriv(rhoE, Q0, Q0)
        Q0_dK0dx_Q0 = self.fltr.apply_gradient(Q0_dK0dx_Q0, x)

        pu2_dR1dx = pu2_pR1px + (pu12_dG0dx_Q0 - mu * pu12_dK0dx_Q0) - 0.5 * pmu2 * Q0_dK0dx_Q0

        R1 = (G0r - mu * K0r) @ u2r + nu * K0r @ u1r - mu * (0.5 * K1r + G1r) @ u1r

        dR1dx_cs = np.zeros((lr, x.size))
        pu2_pR1px_cs = np.zeros(x.size)

        for i in range(x.size):
            x1 = x.copy().astype(complex)
            x1[i] += 1j * dh

            rho1 = self.fltr.apply(x1)
            rhoE1 = np.mean(rho1[self.conn[:, :4]], axis=1)

            l0_1 = l0 + dl0dx[i] * 1j * dh
            mu_1 = -1.0 / l0_1
            Q0_1 = Q0 + dQ0dx[:, i] * 1j * dh
            u0_1 = u0 + du0dx[:, i] * 1j * dh
            u1_1 = u1 + dQ0dx[:, i] * 1j * dh / Q0_norm
            u1r_1 = self.reduce_vector(u1_1)
            Q0r_1 = self.reduce_vector(Q0_1)

            G0_1 = self.get_stress_stiffness_matrix(rho1, u0_1)
            G1_1 = self.get_stress_stiffness_matrix(rho1, u1_1)     
            K0_1 = self.get_stiffness_matrix(rho1)
            K1_1 = self.get_K1(rhoE1, u1_1)
            K11_1 = self.get_K11(rhoE1, u1_1)

            G0r_1 = self.reduce_matrix(G0_1)
            G1r_1 = self.reduce_matrix(G1_1)
            K0r_1 = self.reduce_matrix(K0_1)
            K1r_1 = self.reduce_matrix(K1_1) 
            K11r_1 = self.reduce_matrix(K11_1)

            R1_1 = (G0r_1 - mu_1 * K0r_1) @ u2r + nu * K0r_1 @ u1r_1 - mu_1 * (0.5 * K1r_1 + G1r_1) @ u1r_1

            G0_1 = self.get_stress_stiffness_matrix(rho1, u0_1)
            G1_1 = self.get_stress_stiffness_matrix(rho1, u1)     
            K0_1 = self.get_stiffness_matrix(rho1)
            K1_1 = self.get_K1(rhoE1, u1)
            K11_1 = self.get_K11(rhoE1, u1)

            G0r_1 = self.reduce_matrix(G0_1)
            G1r_1 = self.reduce_matrix(G1_1)
            K0r_1 = self.reduce_matrix(K0_1)
            K1r_1 = self.reduce_matrix(K1_1) 
            K11r_1 = self.reduce_matrix(K11_1)

            R1_2 = (G0r_1 - mu * K0r_1) @ u2r + nu * K0r_1 @ u1r - mu * (0.5 * K1r_1 + G1r_1) @ u1r

            dR1dx_cs[:, i] = (R1_1 - R1).imag / dh
            pu2_pR1px_cs[i] = pu2r @ (R1_2 - R1).imag / dh

        # # check if pu2_pR1px equals to pu2_pR1px_cs
        # ic(np.allclose(pu2_pR1px, pu2_pR1px_cs))
        # exit()

        u1_dK0dx_u2 = self.get_stiffness_matrix_deriv(rhoE, u1, u2)
        u1_dK0dx_u2 = self.fltr.apply_gradient(u1_dK0dx_u2, x)

        pu1_dG0dx_Q0 = self.get_stress_stiffness_matrix_xuderiv(rhoE, u0, pu1, Q0)
        pu1_dG0dx_Q0 = self.fltr.apply_gradient(pu1_dG0dx_Q0, x)

        pu1_dK0dx_Q0 = self.get_stiffness_matrix_deriv(rhoE, pu1, Q0)
        pu1_dK0dx_Q0 = self.fltr.apply_gradient(pu1_dK0dx_Q0, x)

        du1dx = dQ0dx / Q0_norm    

        pR12px_cd = u2 @ K0 @ du1dx + u1_dK0dx_u2
        
        rhs = np.hstack([u2r @ K0r, 0])
        psi = sparse.linalg.spsolve(mat.T, -rhs)
        pu13r = psi[:-1]
        pu13 = self.full_vector(pu13r)
        pmu3 = psi[-1]
        
        pu13_dG0dx_Q0 = self.get_stress_stiffness_matrix_xuderiv(rhoE, u0, pu13, Q0)
        pu13_dG0dx_Q0 = self.fltr.apply_gradient(pu13_dG0dx_Q0, x)
        
        pu13_dK0dx_Q0 = self.get_stiffness_matrix_deriv(rhoE, pu13, Q0)
        pu13_dK0dx_Q0 = self.fltr.apply_gradient(pu13_dK0dx_Q0, x)
        
        
        pR12px = pu13_dG0dx_Q0 - mu * pu13_dK0dx_Q0 - 0.5 * pmu3 * Q0_dK0dx_Q0 + u1_dK0dx_u2

        dbdx = (
            pbpx
            # + pu2r @ dR1dx_cs
            + pu2_dR1dx
            + pnu * pR12px
            + (pu1_dG0dx_Q0 - mu * pu1_dK0dx_Q0)
            - 0.5 * pmu * Q0_dK0dx_Q0
        )


def get_dbdx(self, x, rhoE, l0, Q0):
    u0 = self.u
    u1 = self.u1
    u2 = self.u2
    u1r = self.reduce_vector(u1)
    u2r = self.reduce_vector(u2)
    Q0r = self.reduce_vector(Q0)

    K0r = self.reduce_matrix(self.K0)
    G0r = self.reduce_matrix(self.G0)
    K1r = self.reduce_matrix(self.K1)
    G1r = self.reduce_matrix(self.G1)
    K11r = self.reduce_matrix(self.K11)
    G2 = self.get_stress_stiffness_matrix(self.rho, u2)
    G2r = self.reduce_matrix(G2)

    lr = K0r.shape[0]
    mu = -1.0 / l0
    Q0_norm = np.linalg.norm(Q0)
    nu = mu * u1r @ (0.5 * K1r + G1r) @ u1r * Q0_norm**2

    # solve for governing equations R2
    Ar = G0r - mu * K0r
    Lr = (K0r @ u1r).reshape(-1, 1)
    mat = sparse.bmat([[Ar, Lr], [Lr.T, [0]]], format="csc")
    pbpu2 = Q0_norm**2 * u1r @ (2 * K1r + K1r.T)
    rhs = np.hstack([pbpu2, 0])

    psi = sparse.linalg.spsolve(mat.T, -rhs)
    pu2r = psi[:-1]
    pu2 = self.full_vector(pu2r)
    pnu = psi[-1]

    # solve for governing equations R1
    mat = sparse.bmat(
        [[Ar * Q0_norm, -Lr * Q0_norm], [-(Q0_norm**2) * Lr.T, [0]]], format="csc"
    )
    pbpu1 = 2 * Q0_norm**2 * (u2r @ (K1r + K1r.T + G1r) + u1r @ K11r)
    pR21pu1 = nu * K0r - mu * (K1r + K1r.T + G1r)
    pR22pu1 = u2r @ K0r
    pfpu1 = pbpu1 + pu2r @ pR21pu1 + pnu * pR22pu1
    pfpmu = -pu2r @ K0r @ u2r - pu2r @ (0.5 * K1r + G1r) @ u1r
    rhs = np.hstack([pfpu1, pfpmu])

    psi = sparse.linalg.spsolve(mat.T, -rhs)
    pu1r = psi[:-1]
    pu1 = self.full_vector(pu1r)
    pmu = psi[-1]

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


def get_dadx(self, rhoE, l0, Q0):
    
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
    psi1 = -psi[-1]

    dGdx1 = self.get_stress_stiffness_matrix_xuderiv(rhoE, self.u, psi0, Q0)
    dKdx1 = self.get_stiffness_matrix_deriv(rhoE, psi0, Q0)
    dKdx2 = self.get_stiffness_matrix_deriv(rhoE, Q0, Q0)
    dK1dx = self.get_dK1dx(rhoE, Q0, Q0, Q0)

    dadx = (dGdx1 + dKdx1 / l0) - 0.5 * psi1 * dKdx2 + 1.5 * dK1dx

    return dadx


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
    self.profile["koiter time"] += t2 - t1
    ic(t2 - t1)

    return


def get_dbdx(self, x, rhoE, l0, Q0, dl0dx, du0dx, dQ0dx, dh=1e-30):
    u0 = self.u
    u1 = self.u1
    u2 = self.u2
    u1r = self.reduce_vector(u1)
    u2r = self.reduce_vector(u2)
    Q0r = self.reduce_vector(Q0)

    K0r = self.reduce_matrix(self.K0)
    G0r = self.reduce_matrix(self.G0)
    K1r = self.reduce_matrix(self.K1)
    G1r = self.reduce_matrix(self.G1)
    K11r = self.reduce_matrix(self.K11)
    G2 = self.get_stress_stiffness_matrix(self.rho, u2)
    G2r = self.reduce_matrix(G2)

    lr = K0r.shape[0]
    mu = -1.0 / l0
    Q0_norm = np.linalg.norm(Q0)
    nu = mu * u1r @ (0.5 * K1r + G1r) @ u1r * Q0_norm**2

    pbpu2 = Q0_norm**2 * u1r @ (2 * K1r + K1r.T)
    # pbpu2 = self.get_dbdu2_cs(rhoE, l0, Q0)
    # pbpu2 = self.reduce_vector(pbpu2)

    # solve for governing equations R2
    Ar = G0r - mu * K0r
    Lr = (K0r @ u1r).reshape(-1, 1)
    mat = sparse.bmat([[Ar, Lr], [Lr.T, [0]]], format="csc")
    rhs = np.hstack([pbpu2, 0])

    # print the condtition number of the matrix
    print(np.linalg.cond(Ar.todense()))
    print(np.linalg.cond(mat.todense()))
    
    psi = sparse.linalg.spsolve(mat.T, -rhs)
    pu2r = psi[:-1]
    pu2 = self.full_vector(pu2r)
    pnu = psi[-1]
    # pnu = - pbpu2 @ u1r

    pbpx = np.zeros(x.size)
    pR21px_cs = np.zeros((lr, x.size))
    pR22px_cs = np.zeros(x.size)

    b = Q0_norm**2 * (u2r @ K1r @ u1r + 0.5 * u1r @ K11r @ u1r + 2 * u1r @ K1r @ u2r)
    R21 = (G0r - mu * K0r) @ u2r + self.nu * K0r @ u1r - mu * (0.5 * K1r + G1r) @ u1r
    R22 = u2r @ K0r @ u1r

    for i in range(x.size):
        x1 = x.copy().astype(complex)
        x1[i] += 1j * dh

        rho1 = self.fltr.apply(x1)
        rhoE1 = np.mean(rho1[self.conn[:, :4]], axis=1)

        l0_1 = l0 + dl0dx[i] * 1j * dh
        mu_1 = -1.0 / l0_1
        Q0_1 = Q0 + dQ0dx[:, i] * 1j * dh
        u0_1 = u0 + du0dx[:, i] * 1j * dh
        u1_1 = u1 + dQ0dx[:, i] * 1j * dh / Q0_norm
        u1r_1 = self.reduce_vector(u1_1)
        Q0r_1 = self.reduce_vector(Q0_1)

        G0_1 = self.get_stress_stiffness_matrix(rho1, u0_1)
        G1_1 = self.get_stress_stiffness_matrix(rho1, u1_1)     
        K0_1 = self.get_stiffness_matrix(rho1)
        K1_1 = self.get_K1(rhoE1, u1_1)
        K11_1 = self.get_K11(rhoE1, u1_1)

        G0r_1 = self.reduce_matrix(G0_1)
        G1r_1 = self.reduce_matrix(G1_1)
        K0r_1 = self.reduce_matrix(K0_1)
        K1r_1 = self.reduce_matrix(K1_1) 
        K11r_1 = self.reduce_matrix(K11_1)

        b1 = Q0_norm**2 * (u2r @ K1r_1 @ u1r_1 + 0.5 * u1r_1 @ K11r_1 @ u1r_1 + 2 * u1r_1 @ K1r_1 @ u2r)
        R21_1 = (G0r_1 - mu_1 * K0r_1) @ u2r + self.nu * K0r_1 @ u1r_1 - mu_1 * (0.5 * K1r_1 + G1r_1) @ u1r_1

        pbpx[i] = (b1 - b).imag / dh
        pR21px_cs[:, i] = (R21_1 - R21).imag / dh
        pR22px_cs[i] = (u2r @ K0r_1 @ u1r_1 - R22).imag / dh

    dbdx = pbpx + pu2r @ pR21px_cs + pnu * pR22px_cs

    # # solve for governing equations R1
    # mat = sparse.bmat(
    #     [[Ar * Q0_norm, -Lr * Q0_norm], [-(Q0_norm**2) * Lr.T, [0]]], format="csc"
    # )
    # pR21pu1 = nu * K0r - mu * (K1r + K1r.T + G1r)
    # pR22pu1 = u2r @ K0r
    # pfpu1 = pbpu1 + pu2r @ pR21pu1 + pnu * pR22pu1
    # pfpmu = -pu2r @ K0r @ u2r - pu2r @ (0.5 * K1r + G1r) @ u1r
    # rhs = np.hstack([pfpu1, pfpmu])

    # psi = sparse.linalg.spsolve(mat.T, -rhs)
    # pu1r = psi[:-1]
    # pu1 = self.full_vector(pu1r)
    # pmu = psi[-1]

    # # compute the partial derivative pb/px
    # u2_dK1_u1 = self.get_dK1dx(rhoE, Q0, u2, u1)
    # u1_dK1_u2 = self.get_dK1dx(rhoE, Q0, u1, u2)
    # u1_dK11_u1 = self.get_dK11dx(rhoE, Q0, u1, u1)
    # pbpx = Q0_norm**2 * (u2_dK1_u1 + 0.5 * u1_dK11_u1 + 2 * u1_dK1_u2)

    # # compute pR21 / px
    # pu2_dG0dx_u2 = self.get_stress_stiffness_matrix_xuderiv(rhoE, u0, pu2, u2)
    # pu2_dK0dx_u2 = self.get_stiffness_matrix_deriv(rhoE, pu2, u2)
    # pu2_dK0dx_u1 = self.get_stiffness_matrix_deriv(rhoE, pu2, u1)
    # pu2_dK1_u1 = self.get_dK1dx(rhoE, Q0, pu2, u1)
    # pu2_dG1_u1 = self.get_stress_stiffness_matrix_xderiv(rhoE, u1, pu2, u1)
    # pu2_pR21px = (
    #     pu2_dG0dx_u2
    #     - mu * pu2_dK0dx_u2
    #     + nu * pu2_dK0dx_u1
    #     - mu * (0.5 * pu2_dK1_u1 + pu2_dG1_u1)
    # )

    # # compute pR22 / px
    # u1_dK0dx_u2 = self.get_stiffness_matrix_deriv(rhoE, u1, u2)
    # pR22px = u1_dK0dx_u2

    # # compute the partial derivative pR11/px
    # pu1_dG0dx_Q0 = self.get_stress_stiffness_matrix_xuderiv(rhoE, u0, pu1, Q0)
    # pu1_dK0dx_Q0 = self.get_stiffness_matrix_deriv(rhoE, pu1, Q0)
    # pu1_pR11px = pu1_dG0dx_Q0 - mu * pu1_dK0dx_Q0

    # # compute the partial derivative pR12/px
    # Q0_dK0dx_Q0 = self.get_stiffness_matrix_deriv(rhoE, Q0, Q0)
    # pR12px = -0.5 * Q0_dK0dx_Q0

    # # compute the total derivative db/dx
    # dbdx = pbpx + pu2_pR21px + pnu * pR22px + pu1_pR11px + pmu * pR12px

    return dbdx






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

        self.non_design_nodes = self.prob.topo.fltr.non_design_nodes
        self.design_nodes = np.delete(
            np.arange(self.prob.topo.nnodes), self.non_design_nodes
        )

        self.ndvs = self.prob.topo.fltr.num_design_vars #- len(self.non_design_nodes)
        self.ncon = len(args.confs) if isinstance(args.confs, list) else 1

        super().__init__(comm, nvars=self.ndvs, ncon=self.ncon)

        return

    # def _addMat0(self, which, non_design_nodes):
    #     assert which in ["K", "G"]

    #     rho = np.zeros(self.prob.topo.nnodes)
    #     rho[non_design_nodes] = 1.0

    #     if which == "K":
    #         K00 = self.prob.topo.get_stiffness_matrix(rho)
    #         self.prob.topo.set_K00(K00)
    #     elif which == "G":
    #         G00 = self.prob.topo.get_stress_stiffness_matrix(rho)
    #         self.prob.topo.set_G00(G00)

    #     return

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
        self.prob.topo.x[:] = 1.0
        # ic(self.prob.topo.x.shape, self.prob.topo.x[self.design_nodes].shape)
        self.prob.topo.x[self.design_nodes] = x[:]
        self.prob.initialize()

        self.prob.initialize_koiter(self.Q0_norm)
        ic(self.prob.topo.a, self.prob.topo.b)

        # koiter_list = [
        #     "koiter-a",
        #     "koiter-b",
        #     "koiter-al0",
        #     "koiter-lams",
        #     "koiter-nlams",
        # ]

        # if self.args.objf in koiter_list or any(
        #     [conf in koiter_list for conf in self.args.confs]
        # ):
        #     self.prob.initialize_koiter_derivative()

        # Extract the objective function
        if self.args.objf == "ks-buckling":
            self.obj_scale = self.args.scale_ks_buckling
            self.obj = self.prob.get_ks_buckling()

        elif self.args.objf == "compliance":
            self.obj_scale = self.args.scale_compliance
            self.obj = self.prob.get_compliance()

        elif self.args.objf == "compliance-buckling":
            self.obj_scale = 1.0
            self.c_norm = self.prob.get_compliance() / self.args.c0
            self.ks_norm = self.prob.get_ks_buckling() / self.args.ks0
            self.obj = self.args.w * self.c_norm + (1 - self.args.w) * self.ks_norm

        elif self.args.objf == "aggregate-max":
            self.obj_scale = self.args.scale_aggregate_max
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
            self.obj_scale = 100.0
            self.obj = self.prob.get_koiter_ks_lams(self.args.xi)

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

        if "koiter-b" in self.args.confs:
            b = self.prob.get_koiter_b()
            con.append(b / self.args.b_lb - 1.0)

        return fail, self.obj, con

    def evalObjConGradient(self, x, g, A):

        # Evaluate the gradient of the objective function
        if self.args.objf == "ks-buckling":
            g0 = self.prob.get_ks_buckling_derivative()
            # g0 = dks * self.args.scale_ks_buckling

        elif self.args.objf == "compliance":
            g0 = self.prob.get_compliance_derivative()
            # g0 = dc * self.args.scale_compliance

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
            # g0 = self.prob.topo.xb * self.obj_scale

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

        elif self.args.objf == "koiter-nlams":
            g0 = -self.prob.get_koiter_normalized_dlams(self.args.xi)

        g[:] = g0[:] * self.obj_scale

        index = 0
        if "volume" in self.args.confs:
            A0 = -self.prob.get_area_derivative() / self.area_ub
            A[index][:] = A0[self.design_nodes]
            index += 1

        if "compliance" in self.args.confs:
            A0 = -self.prob.get_compliance_derivative() / self.args.c_ub
            A[index][:] = A0[self.design_nodes]
            index += 1

        if "aggregate" in self.args.confs:
            self.prob.initialize_adjoint()
            self.prob.get_eigenvector_aggregate_derivative()
            self.prob.finalize_adjoint()
            A0 = -self.prob.topo.xb / self.args.h_ub
            A[index][:] = A0[self.design_nodes]
            index += 1

        if "ks-buckling" in self.args.confs:
            dks = self.prob.get_ks_buckling_derivative()
            A0 = -dks / (self.args.BLF_ks_lb * self.ks**2)
            A[index][:] = A0[self.design_nodes]
            index += 1

        if "koiter-b" in self.args.confs:
            dbdx = self.prob.get_koiter_db()
            A0 = dbdx / self.args.b_lb
            A[index][:] = A0[self.design_nodes]
            index += 1

        self.logger.write_output()

        # reset sigma
        self.prob.topo.sigma = self.args.sigma_scale * self.prob.topo.lam[0]

        return False


def settings():
    problem = {
        "domain": "rooda",  # "column" or "rooda"
        "objf": "ks-buckling",
        "w": 0.2,  # weight for compliance-buckling
        "c0": 1e-05,  # 1e-5 compliance reference value
        "ks0": 0.06,  # buckling reference value
        "scale_ks_buckling": 100.0,
        "scale_compliance": 1e5,
        "scale_aggregate_max": 1.0,
        "nx": 64,
        "yxratio": 2,
        "ks_rho": 160.0,  # from ferrari2021 paper
        "rho_agg": 100.0,
        "confs": ["volume"],
        "vol_frac_ub": 0.5,
        "BLF_ks_lb": 10.0,
        "b_lb": 0.00008,
        "c_ub": 4.3 * 7.4e-6,
        "h_ub": 1.8,
        "lb": 1e-06,  # lower bound of design variables
        "maxiter": 1000,  # maximum number of iterations
        "E": 1.0,  # Young's modulus
        "nu": 0.3,  # Poisson's ratio
        "density": 1.0,
    }

    koiter = {
        "xi": -1e-3,
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
        "sigma": 10.0,
        "sigma_scale": 1.1,
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
        Lx=8.0,
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
        E=args.E,
        nu=args.nu,
        density=args.density,
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

from icecream import ic
import numpy as np
from sympy import Eq, diff, latex, solve, symbols

# xib = 1e0
# kb = np.abs(1.661e-03)
# cof = [1, 27 / 4 * kb * xib**2 - 3, 3, -1]
# r = np.roots(cof)
# print(r)

# a = (1 - r) ** 1.5 - 3 * np.sqrt(3) / 2 * np.sqrt(kb) * xib * r
# print(a)

# # find the root of the equation whose is real part less than 1 but greater than 0
# lams = np.real(r[np.logical_and(r < 1, r > 0)])[0]
# print(lams)

# a = cof[0]
# b = cof[1]
# c = cof[2]
# d = cof[3]

# p = (3 * a * c - b**2) / (3 * a**2)
# q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)

# D = q**2 / 4 + p**3 / 27
# ic(D)

# if D > 1e-15:
#     x = np.cbrt(-q / 2 + np.sqrt(D)) + np.cbrt(-q / 2 - np.sqrt(D)) - b / (3 * a)
# elif 0 < D < 1e-15:
#     x = 2 * np.cbrt(-q / 2) - b / (3 * a)
# else:
#     theta = np.arccos(-q / 2 * np.sqrt(-27 / p**3))
#     x = 2 * np.sqrt(-p / 3) * np.cos(theta / 3) - b / (3 * a)

# print(x)


def fx(a, b, c, d, dp=None, dq=None, db=None):
    if db is not None:
        b += db
    ic(a, b, c, d)
    p = (3 * a * c - b**2) / (3 * a**2)
    q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)

    if dp is not None:
        p += dp

    if dq is not None:
        q += dq

    D = q**2 / 4 + p**3 / 27
    ic(D)

    if D > 0:
        a1 = -q / 2 + np.sqrt(D)
        a2 = -q / 2 - np.sqrt(D)
        x = np.cbrt(a1) + np.cbrt(a2) - b / (3 * a)
    else:
        theta = np.arccos(-q / 2 * np.sqrt(-27 / p**3))
        x = 2 * np.sqrt(-p / 3) * np.cos(theta / 3) - b / (3 * a)
    ic(x)
    return x


def df_dq(a, b, c, d):
    p = (3 * a * c - b**2) / (3 * a**2)
    q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)

    D = q**2 / 4 + p**3 / 27

    if D > 0:
        a1 = -q / 2 + np.sqrt(D)
        a2 = -q / 2 - np.sqrt(D)

        # Derivative of discriminant D with respect to q
        dD_dq = q / 2

        # Derivatives of a1 and a2 with respect to q
        da1_dq = -1 / 2 + (1 / 2) * (dD_dq / np.sqrt(D))
        da2_dq = -1 / 2 - (1 / 2) * (dD_dq / np.sqrt(D))

        # Applying the chain rule to cube roots
        b1 = (1 / (3 * np.cbrt(a1) ** 2)) * da1_dq
        b2 = (1 / (3 * np.cbrt(a2) ** 2)) * da2_dq

        df_dq = b1 + b2
    else:
        a1 = -q / 2 * np.sqrt(-27 / p**3)
        theta = np.arccos(a1)

        dtheta_da1 = -1 / np.sqrt(1 - a1**2)
        da1_dq = -1 / 2 * np.sqrt(-27 / p**3)
        dtheta_dq = dtheta_da1 * da1_dq

        dxdtheta = 2 * np.sqrt(-p / 3) * (-np.sin(theta / 3) / 3)
        df_dq = dxdtheta * dtheta_dq

    return df_dq


def df_dp(a, b, c, d):
    p = (3 * a * c - b**2) / (3 * a**2)
    q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)

    D = q**2 / 4 + p**3 / 27

    if D > 0:
        a1 = -q / 2 + np.sqrt(D)
        a2 = -q / 2 - np.sqrt(D)

        # Derivative of discriminant D with respect to p
        dD_dp = (p**2) / 9

        # Derivatives of a1 and a2 with respect to p
        da1_dp = (1 / 2) * (dD_dp / np.sqrt(D))
        da2_dp = -(1 / 2) * (dD_dp / np.sqrt(D))

        # Applying the chain rule to cube roots
        b1 = (1 / (3 * np.cbrt(a1**2))) * da1_dp
        b2 = (1 / (3 * np.cbrt(a2**2))) * da2_dp

        df_dp = b1 + b2
    else:
        a1 = -q / 2 * np.sqrt(-27 / p**3)
        theta = np.arccos(a1)

        dtheta_da1 = -1 / np.sqrt(1 - a1**2)
        da1_dp = -q / 2 * (1 / 2) / np.sqrt(-27 / p**3) * 81 * p ** (-4)
        dtheta_dp = dtheta_da1 * da1_dp

        dxdtheta = 2 * np.sqrt(-p / 3) * (-np.sin(theta / 3) / 3)
        df_dp0 = 1 / np.sqrt(-p / 3) * (-1 / 3) * np.cos(theta / 3)
        df_dp = dxdtheta * dtheta_dp + df_dp0

    return df_dp


def dfx_db(a, b, c, d):
    dfdp = df_dp(a, b, c, d)
    dfdq = df_dq(a, b, c, d)

    dpdb = -2 * b / (3 * a**2)
    dqdb = (6 * b**2 - 9 * a * c) / (27 * a**3)

    return dfdp * dpdb + dfdq * dqdb - 1 / (3 * a)


def dfx_dkb(a, b, c, d, xib, kb):

    if kb < 0:
        dbdkb = -27 / 4 * xib**2
    else:
        dbdkb = 27 / 4 * xib**2
    return dfx_db(a, b, c, d) * dbdkb


xib = 1e-3
kb = 1.661e-03

if kb < 0:
  cof = [1, -27 / 4 * kb * xib**2 - 3, 3, -1]
else:
  cof = [1, 27 / 4 * kb * xib**2 - 3, 3, -1]
  
r = np.roots(cof)
print(r)

a = cof[0]
b = cof[1]
c = cof[2]
d = cof[3]

x = fx(a, b, c, d)
ic(x)

# dq = 1e-4
# dp = 1e-5

# x1 = fx(a, b, c, d, dq=dq)
# x2 = fx(a, b, c, d, dq=-dq)
# dx_numeric = (x1 - x2) / (2 * dq)
# dx_analytic = df_dq(a, b, c, d)

# print("FD  q:", dx_numeric)
# print("ANS q:", dx_analytic)
# print("err q:", (dx_numeric - dx_analytic) / dx_analytic)

# # Finite difference for dp
# x1_dp = fx(a, b, c, d, dp=dp)
# x2_dp = fx(a, b, c, d, dp=-dp)
# dx_numeric_dp = (x1_dp - x2_dp) / (2 * dp)
# dx_analytic_dp = df_dp(a, b, c, d)

# print("FD  p:", dx_numeric_dp)
# print("ANS p:", dx_analytic_dp)
# print("err p:", (dx_numeric_dp - dx_analytic_dp) / dx_analytic_dp)

# db = 1e-6
# x1 = fx(a, b, c, d, db=db)
# x2 = fx(a, b, c, d, db=-db)
# dx_numeric = (x1 - x2) / (2 * db)
# dx_analytic = dfx_db(a, b, c, d)

# print("FD  b:", dx_numeric)
# print("ANS b:", dx_analytic)
# print("err b:", (dx_numeric - dx_analytic) / dx_analytic)

dh = 1e-6
kb1 = kb + dh

if kb1 < 0:
  b1 = -27 / 4 * kb1 * xib**2 - 3
else:
  b1 = 27 / 4 * kb1 * xib**2 - 3

kb2 = kb - dh

if kb2 < 0:
  b2 = -27 / 4 * kb2 * xib**2 - 3
else:
  b2 = 27 / 4 * kb2 * xib**2 - 3

x1 = fx(a, b1, c, d)
x2 = fx(a, b2, c, d)
dx_numeric = (x1 - x2) / (2 * dh)
dx_analytic = dfx_dkb(a, b, c, d, xib, kb)

print("FD  kb:", dx_numeric)
print("ANS kb:", dx_analytic)
print("err kb:", (dx_numeric - dx_analytic) / dx_analytic)

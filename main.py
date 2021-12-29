import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate


def newton(x, y, x0):
    A0 = y[0]
    A01 = (y[0] - y[1]) / (x[0] - x[1])
    A12 = (y[1] - y[2]) / (x[1] - x[2])
    A23 = (y[2] - y[3]) / (x[2] - x[3])
    A012 = (A01 - A12) / (x[0] - x[2])
    A123 = (A12 - A23) / (x[1] - x[3])
    A0123 = (A012 - A123) / (x[0] - x[3])
    return A0 + A01 * (x0 - x[0]) + A012 * (x0 - x[0]) * (x0 - x[1]) + A0123 * (x0 - x[0]) * (x0 - x[1]) * (x0 - x[2])


def lagrange(x, y, x0):
    fi1 = (x0 - x[1]) * (x0 - x[2]) * (x0 - x[3]) / ((x[0] - x[1]) * (x[0] - x[2]) * (x[0] - x[3]))

    fi2 = (x0 - x[0]) * (x0 - x[2]) * (x0 - x[3]) / ((x[1] - x[0]) * (x[1] - x[2]) * (x[1] - x[3]))

    fi3 = (x0 - x[0]) * (x0 - x[1]) * (x0 - x[3]) / ((x[2] - x[0]) * (x[2] - x[1]) * (x[2] - x[3]))

    fi4 = (x0 - x[0]) * (x0 - x[1]) * (x0 - x[2]) / ((x[3] - x[0]) * (x[3] - x[1]) * (x[3] - x[2]))

    a1 = y[0] * fi1
    a2 = y[1] * fi2
    a3 = y[2] * fi3
    a4 = y[3] * fi4
    return a1 + a2 + a3 + a4


def newton_poly(x, newP, newC):
    b0, b1, b2, b3 = newC[0], newC[1], newC[2], newC[3]
    x0, x1, x2 = newP[0], newP[1], newP[2]
    return b0 + b1 * (x - x0) + b2 * (x - x0) * (x - x1) + b3 * (x - x0) * (x - x1) * (x - x2)


def search_square_polynomial_coefficients(x, y, m):
    xy = []
    for k in range(len(x)):
        xy.append([x[k], y[k]])
    matrix, r_side_of_matrix = MakeSystem(xy, m)
    cPoly = np.linalg.solve(matrix, r_side_of_matrix)
    return cPoly


def MakeSystem(xy_list, basis):
    matrix = [[0] * basis for _ in range(basis)]
    right_side_of_matrix = [0] * basis
    for i in range(basis):
        for j in range(basis):
            sumA, sumB = 0, 0
            for k in range(len(xy_list)):
                sumA += xy_list[k][0] ** (i + j)
                sumB += xy_list[k][1] * xy_list[k][0] ** i
            matrix[i][j] = sumA
            right_side_of_matrix[i] = sumB
    return matrix, right_side_of_matrix


def create_square_polynomial(c, x):
    polynomial = 0
    for i in range(len(c)):
        polynomial += c[i] * x ** i
    return polynomial


def search_coefficients_SLAE(x, y, m):
    xy = []
    for k in range(len(x)):
        xy.append([x[k], y[k]])
    matrix, r_side_of_matrix = make_system_SLAE(xy, m)
    coeff_poly = np.linalg.solve(matrix, r_side_of_matrix)
    return coeff_poly


def make_system_SLAE(points, basis):
    matrix = [[0] * basis for _ in range(basis)]
    right_side_of_matrix = [0] * basis
    for i in range(basis):
        for j in range(basis):
            matrix[i][j] = points[i][0] ** j
        right_side_of_matrix[i] = points[i][1]
    return matrix, right_side_of_matrix


args = [0, 1, 2, 3]
Y = [0, 6, 3, 5]
coefficients = [0.955, -0.687, -0.150, 0.352]
X = np.arange(0, 3.01, 0.01)

plt.title('interpolate')
plt.plot(X, lagrange(args, Y, X), '--', label='lagrange')
plt.plot(X, newton(args, Y, X), 'g-', label='newton')

tck = scipy.interpolate.splrep(args, Y)
sp_y = scipy.interpolate.splev(args, tck)
plt.plot(args, sp_y, label='spline')

poly = search_square_polynomial_coefficients(args, Y, 5)
plt.plot(X, create_square_polynomial(poly, X), 'b--', label='Square interpolation (m=5)')
poly_SLAE = search_coefficients_SLAE(args, Y, 4)
plt.plot(X, create_square_polynomial(poly_SLAE, X), 'r--', label='SLAE interpolation')
plt.grid()
plt.legend()
plt.show()

new_X = np.arange(0, 5.01, 0.01)
plt.title('extrapolate')
f = scipy.interpolate.interp1d(args, Y, fill_value='extrapolate')
plt.plot(new_X, f(new_X), label='spline extrapolate')
plt.plot(new_X, lagrange(args, Y, new_X), '--', label='lagrange')
plt.plot(new_X, newton(args, Y, new_X), 'g-', label='newton')
plt.plot(new_X, create_square_polynomial(poly, new_X), 'b--', label='Square')
plt.plot(new_X, create_square_polynomial(poly_SLAE, new_X), 'r--', label='SLAE')
plt.grid()
plt.legend()
plt.show()

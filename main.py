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


def search_square_polynomial_coefficients(x, y, m):
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

    xy = []
    for k in range(len(x)):
        xy.append([x[k], y[k]])
    matrix, r_side_of_matrix = MakeSystem(xy, m)
    cPoly = np.linalg.solve(matrix, r_side_of_matrix)
    return cPoly


def create_polynomial(c, x):
    polynomial = 0
    for i in range(len(c)):
        polynomial += c[i] * x ** i
    return polynomial


def search_coefficients_SLAE(x, y, m):
    def make_system_SLAE(points, basis):
        matrix = [[0] * basis for _ in range(basis)]
        right_side_of_matrix = [0] * basis
        for i in range(basis):
            for j in range(basis):
                matrix[i][j] = points[i][0] ** j
            right_side_of_matrix[i] = points[i][1]
        return matrix, right_side_of_matrix

    xy = []
    for k in range(len(x)):
        xy.append([x[k], y[k]])
    matrix, r_side_of_matrix = make_system_SLAE(xy, m)
    coeff_poly = np.linalg.solve(matrix, r_side_of_matrix)
    return coeff_poly


def parabola_method(func, n, h):
    a = 0
    b = 3
    integral = func(a) + func(b)
    x = a
    for i in range(1, n):
        x += h
        if i % 2:
            integral += 4 * func(x)
        else:
            integral += 2 * func(x)
    return h / 3 * integral


def derivative(func, power):
    n = 2
    a = 0
    b = 3
    h = (b - a) / n
    while (abs(parabola_method(func, n, h / 2) - parabola_method(func, n, h))) > 10 ** power and n < 2000:
        n += 1
        if n % 2:
            n += 1
        h = (b - a) / n
    return round(parabola_method(func, n, h), abs(power) + 1)


def central_diff_ratio(m, x, y):
    i = 0
    for j in range(1, len(x) - 1):
        if x[j - 1] <= m < x[j + 1]:
            i = j
            break
    return 0.5 * (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i])


X = [0, 1, 2, 3]
Y = [0, 6, 3, 5]
point = 2.34
args = np.arange(0, 3.01, 0.01)
eps = 1e-3
power = -3

plt.title('interpolate')
plt.plot(args, lagrange(X, Y, args), '--', label='lagrange')
plt.plot(args, newton(X, Y, args), 'g-', label='newton')

tck = scipy.interpolate.splrep(X, Y)
sp_y = scipy.interpolate.splev(args, tck)
plt.plot(args, sp_y, label='spline')

poly = search_square_polynomial_coefficients(X, Y, 5)
plt.plot(args, create_polynomial(poly, args), 'b--', label='Square interpolation (m=5)')
poly_SLAE = search_coefficients_SLAE(X, Y, 4)
plt.plot(args, create_polynomial(poly_SLAE, args), 'r--', label='SLAE interpolation')
plt.grid()
plt.legend()
plt.show()

new_args = np.arange(0, 5.01, 0.01)
plt.title('extrapolate')
spline = scipy.interpolate.interp1d(X, Y, fill_value='extrapolate')
plt.plot(new_args, spline(new_args), label='spline extrapolate')
plt.plot(new_args, lagrange(X, Y, new_args), '--', label='lagrange')
plt.plot(new_args, newton(X, Y, new_args), 'g-', label='newton')
plt.plot(new_args, create_polynomial(poly, new_args), 'b--', label='Square')
plt.plot(new_args, create_polynomial(poly_SLAE, new_args), 'r--', label='SLAE')
plt.grid()
plt.legend()
plt.show()

print('\tPoint x = ', point)
print('Lagrange ', lagrange(X, Y, point))
print('Newton ', newton(X, Y, point))
print('Square ', create_polynomial(poly, point))
print('SLAE ', create_polynomial(poly_SLAE, point))
print('Spline ', scipy.interpolate.splev(point, tck))

print('\n\tIntegral:')
print('Lagrange ', derivative(lambda x: lagrange(X, Y, x), power))
print('Newton ', derivative(lambda x: newton(X, Y, x), power))
print('Square ', derivative(lambda x: create_polynomial(poly, x), power))
print('SLAE ', derivative(lambda x: create_polynomial(poly_SLAE, x), power))
print('Spline ', derivative(lambda x: scipy.interpolate.splev(x, tck), power))

print('\n\tDerivative x = ', point)
print('Lagrange ', central_diff_ratio(point, args, lagrange(X, Y, args)))
print('Newton ', central_diff_ratio(point, args, newton(X, Y, args)))
print('Square ', central_diff_ratio(point, args, create_polynomial(poly, args)))
print('SLAE ', central_diff_ratio(point, args, create_polynomial(poly_SLAE, args)))
print('Spline ', central_diff_ratio(point, args, sp_y))

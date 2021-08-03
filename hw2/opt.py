#!/usr/bin/env python3

from collections import namedtuple
import numpy as np

Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Результаты оптимизации

Attributes
----------
nfev : int
    Полное число вызовов модельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива — менее nfev
gradnorm : float
    Норма градиента на финальном итерационном шаге
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""


def gauss_newton(y, f, j, x0, k=1, tol=1e-4):
    i = 0 
    x = x0
    cost = []
    r = (y - f(x))
    delta_x = r
    while np.linalg.norm(delta_x) > tol * np.linalg.norm(x):
        jac = j(x)
        i += 1
        g = np.dot(jac.T, r)
        delta_x = np.linalg.solve(jac.T @ jac, g)
        x = x + k* delta_x
        r = (y - f(x))
        cost.append(0.5 * r @ r)
    return Result(nfev=i, cost=cost, gradnorm=np.linalg.norm(g), x=x)

def lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4):
    x = x0
    lmbd = lmbd0
    i = 0
    delta_x = y - f(x)
    g = y - f(x)
    cost = []
    while np.linalg.norm(delta_x) > tol * np.linalg.norm(x):
        jac = j(x)
        F_pre = g.T @ g
        dx = np.linalg.solve(jac.T @ jac + lmbd * np.identity(len(x)), jac.T @ g)
        F = (y - f(x + dx)).T @ (y - f(x + dx))
        dx_nu = np.linalg.solve(jac.T @ jac + lmbd / nu * np.identity(len(x)), jac.T @ g)
        F_nu = (y - f(x + dx_nu)).T @ (y - f(x + dx_nu))
        
        if F_nu <= F_pre:
            x += dx_nu
            delta_x = dx_nu
            lmbd = lmbd / nu
        elif F_nu > F_pre and F_nu <= F:
            x += dx
            delta_x = dx
        else:
            F_w = F_pre + 1
            w = 2
            while F_w > F_pre and w < 100:
                dx_w = np.linalg.solve(jac.T @ jac + lmbd * nu * w * np.identity(len(x)), jac.T @ g)
                w += 1
                F_w = (y - f(x + dx_w)) @ (y - f(x + dx_w))
            x += dx_w
            delta_x = dx_w
            lmbd = lmbd * w
        g = y - f(x)
        cost.append(0.5 * g @ g)
        i += 1
    return Result(nfev=i, cost=cost, gradnorm=np.linalg.norm(jac.T @ g), x=x)


if __name__ == "__main__":
    pass

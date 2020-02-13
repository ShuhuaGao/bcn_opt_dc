"""
Biological example: ara operon network
"""
from algorithm.bcn import BooleanControlNetwork
from algorithm.proposed import MadaniSolver, ValueIterationSolver
from algorithm.related_work import ChengSolver, ZhuSolver
from algorithm.utils import read_network
import time


def inverse_map(i: int, n: int):
    """
    Accumulative STP of logical variables is bijective.
    Given a result i (\delta_{2^n}^i), find the corresponding logical values.

    :return a list of 0/1
    """
    r = []
    while n > 0:
        if i % 2 == 0:
            r.append(0)
            i = i // 2
        else:
            r.append(1)
            i = (i + 1) // 2
        n = n - 1
    r.reverse()
    return r


def g(i, k):
    """
    Stage cost

    :param i: state
    :param k: control
    :return: the cost
    """
    n = 9
    m = 4
    X = inverse_map(i, n)
    U = inverse_map(k, m)
    A = [-28, -12, 12, 16, 0, 0, 0, 20, 16]
    B = [-8, 40, 20, 40]
    return sum(a * x for a, x in zip(A, X)) + sum(b * u for b, u in zip(B, U))


def run(solver_class, name, bcn, g, lamb, x0, theta=None):
    print(f'Running method [{name}]...')
    ts = time.time()
    if theta is None:
        solver = solver_class(bcn, g, lamb)
    else:
        solver = solver_class(bcn, g, lamb, theta)
    solver.solve(x0)
    if theta is None:
        print(f'Method [{name}]: time = {time.time() - ts: .2f} s')
    else:
        print(f'Method [{name}] (theta = {theta}): time = {time.time() - ts: .2f} s')


if __name__ == '__main__':
    n, m, L = read_network('./networks/ara_operon.txt')
    bcn = BooleanControlNetwork(n, m, L)
    lamb = 0.6
    x0 = 10
    # madani_solver = MadaniSolver(bcn, g, lamb)
    # madani_opt, madani_K = madani_solver.solve(x0)
    # print(madani_opt)
    run(MadaniSolver, 'Madani', bcn, g, lamb, x0)
    for theta in [0.1, 0.01, 0.001]:
        run(ValueIterationSolver, 'Value iteration', bcn, g, lamb, x0, theta)
    print("NOTE: the two existing methods will run for hours")
    run(ZhuSolver, 'Zhu', bcn, g, lamb, x0)
    run(ChengSolver, 'Cheng', bcn, g, lamb, x0)




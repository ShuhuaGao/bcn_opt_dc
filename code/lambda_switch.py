"""
A simple lambda switch network (see Example 2 in [1])
[1] Q. Zhu, Y. Liu, J. Lu, and J. Cao, “On the optimal control of boolean
control networks,” SIAM Journal on Control and Optimization, vol. 56,
no. 2, pp. 1321–1341, 2018.

Mainly to verify the correctness of the algorithm implementations, i.e., all four algorithms
produce the same results.
"""
from algorithm.bcn import BooleanControlNetwork
from algorithm.proposed import MadaniSolver, ValueIterationSolver
from algorithm.related_work import ChengSolver, ZhuSolver
from algorithm.utils import read_network
import numpy as np

# np.random.seed(16)
Q = np.random.uniform(-5, 30, 32)
R = [8, 10]


def g(i, k):
    """
    Stage cost

    :param i: state
    :param k: control
    :return: the cost
    """
    return Q[i - 1] + R[k - 1]


if __name__ == '__main__':
    n, m, L = read_network('./networks/lambda_switch.txt')
    bcn = BooleanControlNetwork(n, m, L)
    lamb = 0.6
    x0 = 10
    madani_solver = MadaniSolver(bcn, g, lamb)
    madani_opt, madani_K = madani_solver.solve(x0)
    print('madani: ', madani_opt)
    cheng_solver = ChengSolver(bcn, g, lamb)
    cheng_opt = cheng_solver.solve(x0)
    print('cheng: ', cheng_opt)
    vi_solver = ValueIterationSolver(bcn, g, lamb, theta=0.001)
    vi_opt, vi_K = vi_solver.solve(x0)
    print('value iteration: ', vi_opt)
    zhu_solver = ZhuSolver(bcn, g, lamb)
    zhu_opt = zhu_solver.solve(x0)
    print('zhu: ', zhu_opt)

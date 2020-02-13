"""
Existing work
[1]  D. Cheng, Y. Zhao, and J.-B. Liu, “Optimal control of finite-valued
networks,” Asian Journal of Control, vol. 16, no. 4, pp. 1179–1190,
2014.
[2]  Q. Zhu, Y. Liu, J. Lu, and J. Cao, “On the optimal control of boolean
control networks,” SIAM Journal on Control and Optimization, vol. 56,
no. 2, pp. 1321–1341, 2018.

"""
from typing import Iterable, Tuple, List, Union, Dict, Callable
from .stp import *
from . import bool_algebra as ba
from .proposed import DiscountedCostSolver
from .bcn import BooleanControlNetwork
import numpy as np
import copy
from operator import itemgetter
import time


class ChengSolver(DiscountedCostSolver):
    """
    Algorithm developed in [1]
    """
    def __init__(self, bcn: BooleanControlNetwork, g: Callable[[int, int], float], lamb: float,
                 Cx: Iterable[int] = None, Cu: Callable[[int], Iterable[int]] = None):
        """
        Initialize the solver.

        :param bcn: a Boolean control network
        :param g: stage cost function, g(x, u) --> cost
        :param Cx: state constraints, only states in Cx are allowed
        :param Cu: control constraints, Cu(i) gives the control that are allowed at i
        """
        super().__init__(bcn, g, lamb, Cx, Cu)
        self._cm = self._compute_cost_matrix()   # 1-step cost matrix

    def _compute_cost_matrix(self):
        # M1 in the paper
        # CM[i, j] is the min cost from i to j in one step
        cm = np.full((self.N + 1, self.N + 1), np.inf)
        for k in range(1, self.M + 1):
            for i in range(1, self.N + 1):
                blk = self.L[(k - 1) * self.N: k * self.N]
                j = blk[i - 1]
                # i --> j with control k
                cm[i][j] = min(cm[i][j], self.g(i, k))
        return cm

    def solve(self, x0: int) -> float:
        """
        Solve the problem and return the optimal performance index

        :param x0: initial state
        :return:
        """
        # phase 1
        # print('phase 1')
        M = [None] * (self.N + 1)
        M[1] = self._cm
        for l in range(2, self.N + 1):
            print(f'l = {l}')
            ts = time.time()
            Ml = np.empty((self.N + 1, self.N + 1))
            for i in range(1, self.N + 1):
                for j in range(1, self.N + 1):
                    # i --> b in (l-1) steps and b --> j in 1 step
                    Ml[i, j] = min(self.lamb ** (l - 1) * M[1][b, j] + M[l - 1][i, b]
                                   for b in range(1, self.N + 1))
            M[l] = Ml
            print(f'time = {time.time() - ts: .2f}')
        c_star = [None] * (self.N + 1)  # optimal circle
        for i in range(1, self.N + 1):
            c_star[i] = min(M[l][i, i] / (1 - self.lamb ** l) for l in range(1, self.N + 1))
        # print('phase 2')
        # print('c*: ', c_star[13], c_star[24], c_star[15], c_star[26])
        # phase 2
        # given x0, suppose the entry to the optimal cycle is i, test each possible i and l
        opt = np.inf
        for l in range(1, self.N + 1):
            for i in range(1, self.N + 1):
                # x0 reach i in l steps
                opt = min(opt, M[l][x0, i] + self.lamb ** l * c_star[i])
        return opt


class ZhuSolver(DiscountedCostSolver):
    """
        Algorithm developed in [2]
    """

    def __init__(self, bcn: BooleanControlNetwork, g: Callable[[int, int], float], lamb: float,
                 Cx: Iterable[int] = None, Cu: Callable[[int], Iterable[int]] = None):
        """
        Initialize the solver.

        :param bcn: a Boolean control network
        :param g: stage cost function, g(x, u) --> cost
        :param Cx: state constraints, only states in Cx are allowed
        :param Cu: control constraints, Cu(i) gives the control that are allowed at i
        """
        super().__init__(bcn, g, lamb, Cx, Cu)
        # compute the control and cost matrix
        self.D = np.full([self.N + 1, self.N + 1], np.inf)
        self.C = np.zeros((self.N + 1, self.N + 1), dtype=np.uint64)
        for i in range(1, self.N + 1):
            for k in range(1, self.M + 1):
                # i --> j with control k
                blk = self.L[(k - 1) * self.N: k * self.N]
                j = blk[i - 1]
                cik = self.g(i, k)
                if cik < self.D[i][j]:
                    self.D[i][j] = cik
                    self.C[i][j] = k

    def _circle(self, A, B, df):
        """
        The circle operator.

        :param A: n-by-n matrix
        :param B: n-by-n matrix
        :return: a n-by-n matrix
        """
        assert A.shape == B.shape
        n = A.shape[0] - 1
        C = A.copy()
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                C[i][j] = min(A[i][k] + df * B[k][j] for k in range(1, n + 1))
        return C

    def _stp(self, i, p, j, q):
        """
        STP of \delta_p^i and \delta_q^j

        :return: t of \delta_{p*q}^t
        """
        if i is None or j is None:
            return None
        return (i - 1) * q + j

    def solve(self, x0: int) -> float:
        M = self.M
        N = self.N
        C = [None] * (N + 1)
        D = [None] * (N + 1)
        P = np.full(N + 1, np.inf)
        # algorithm 3
        C[1] = self.C
        D[1] = self.D
        for i in range(1, N + 1):
            P[i] = min(P[i], D[1][i, i] / (1 - self.lamb))
        for s in range(2, N + 1):
            # print(f's = {s}')
            # ts = time.time()
            D[s] = self._circle(D[s - 1], self.D, self.lamb ** (s - 1))
            Cs = np.zeros((self.N + 1, self.N + 1), dtype=np.uint32)
            for i in range(1, N + 1):
                for j in range(1, N + 1):
                    if D[s][i, j] == np.inf:
                        Cs[i, j] = 0
                    else:
                        _, k_star = min(((D[s-1][i, k] + self.lamb ** (s - 1) * self.D[k, j], k)
                                        for k in range(1, N + 1)), key=itemgetter(0))
                        Cs[i, j] = self._stp(C[s-1][i, k_star], M**(s-1), self.C[k_star, j], M)
            C[s] = Cs
            # cycle
            for i in range(1, N + 1):
                if D[s][i, i] < np.inf:
                    P[i] = min(P[i], D[s][i, i] / (1 - self.lamb ** s))
            # print(f'time = {time.time() - ts: .2f}')
        # algorithm 4
        # print('P: ', P[13], P[24], P[15], P[26])
        opt = np.inf
        for j in range(1, N + 1):
            for s in range(1, N + 1):
                opt = min(opt, D[s][x0, j] + self.lamb ** s * P[j])
        return opt









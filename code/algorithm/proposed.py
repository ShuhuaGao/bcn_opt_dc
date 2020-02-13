"""
Proposed methods for "Optimal Control of Boolean Control Networks with Discounted Cost: An
Efficient Approach based on Deterministic Markov Decision Process"
(1) Value iteration
(2) Madani's algorithm
"""
from typing import Iterable, Tuple, List, Union, Dict, Callable
import networkx as nx
from operator import itemgetter
from .bcn import BooleanControlNetwork

import time


class DiscountedCostSolver:
    """
    Base class for solvers
    """

    def __init__(self, bcn: BooleanControlNetwork, g: Callable[[int, int], float], lamb: float,
                 Cx: Iterable[int] = None, Cu: Callable[[int], Iterable[int]] = None):
        """
        Initialize the solver.

        :param bcn: a Boolean control network
        :param g: stage cost function, g(x, u) --> cost
        :param lamb: discount factor
        :param Cx: state constraints, only states in Cx are allowed
        :param Cu: control constraints, Cu(i) gives the control that are allowed at i
        """
        self.bcn = bcn
        assert 0 < lamb < 1
        self.lamb = lamb
        self.M = self.bcn.M
        self.N = self.bcn.N
        self.L = self.bcn.L
        if Cx is None:
            self.Cx = set(i for i in range(1, self.N + 1))
        else:
            if not isinstance(Cx, set):
                Cx = set(Cx)
            self.Cx = Cx
        self.Cu = Cu
        self.g = g

    def solve(self, x0: int) -> Tuple[float, Dict[int, int]]:
        """
        Solve the infinite-horizon optimal control problem.

        :return: the optimal value of the index function and the state feedback matrix
        """
        pass

    def _is_input_allowed(self, i, u):
        if self.Cu is None:
            return True
        return u in self.Cu(i)


class MadaniSolver(DiscountedCostSolver):
    """
    Madani's algorithm.
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
        self._stg: nx.DiGraph = None

    def build_stg(self) -> nx.DiGraph:
        """
        Build the STG for a given Boolean control network subject to constraints.

        :return: a weighted directed graph
        """
        L = self.L
        stg = nx.DiGraph()
        N = self.N
        M = self.M
        stg.add_nodes_from(self.Cx)
        for k in range(M):
            blk = L[k * N: (k + 1) * N]
            for i in self.Cx:
                if not self._is_input_allowed(i, k):
                    continue
                j = blk[i - 1]  # state i -> j by control k
                if j not in self.Cx:
                    continue  # but the successor state j is not allowed
                stg.add_edge(i, j)
                if 'Uij' in stg.edges[i, j]:
                    stg.edges[i, j]['Uij'].append(k + 1)
                else:
                    stg.edges[i, j]['Uij'] = [k + 1]
        # after we build the graph topology, now compute the cost
        for i in stg:
            for j in stg.adj[i]:
                action = min(((self.g(i, k), k) for k in stg.edges[i, j]['Uij']),
                             key=itemgetter(0))
                stg.edges[i, j]['weight'] = action[0]
                stg.edges[i, j]['u'] = action[1]
        self._stg = stg
        return stg

    def solve(self, x0: int) -> Tuple[float, Dict[int, int]]:
        """
        Solve the infinite-horizon optimal control problem.

        :return: the optimal value of the index function and the state feedback matrix
        """
        if self._stg is None:
            self.build_stg()
        nv = len(self.Cx)
        V = self.Cx
        # compute d_k
        d = [{} for _ in range(nv + 1)]
        for i in V:
            d[0][i] = 0
        for k in range(1, nv + 1):
            for i in V:
                d[k][i] = min(self._stg.edges[i, j]['weight'] + self.lamb * d[k - 1][j]
                              for j in self._stg.adj[i])

        # compute y0
        def compute_y0(i, k):
            t = self.lamb ** (nv - k)
            return (d[nv][i] - t * d[k][i]) / (1 - t)

        y0 = {}
        for i in V:
            y0[i] = max(compute_y0(i, k) for k in range(nv))

        # compute yk
        y = [{} for _ in range(nv)]
        y[0] = y0
        for k in range(1, nv):
            for i in V:
                y[k][i] = min(self._stg.edges[i, j]['weight'] + self.lamb * y[k - 1][j]
                              for j in self._stg.adj[i])
        # compute v*
        v_star = {i: min(y[k][i] for k in range(nv)) for i in V}
        # compute optimal policy
        pi_star = self._resolve_optimal_policy(v_star)
        # feedback matrix
        K = {}
        for i, (u, _) in pi_star.items():
            K[i] = u
        return pi_star[x0][1], K

    def _resolve_optimal_policy(self, v_star: Dict) -> Dict[int, Tuple[int, float]]:
        """
        Given the optimal value function, determine an optimal policy

        :return: state -> (optimal action, optimal value of infinite-horizon control)
        """
        assert self._stg is not None
        pi = {}
        for i in self.Cx:
            v, j = min(((self._stg.edges[i, j]['weight'] + self.lamb * v_star[j], j)
                        for j in self._stg.adj[i]), key=itemgetter(0))
            pi[i] = (self._stg.edges[i, j]['u'], v)
        return pi


class ValueIterationSolver(DiscountedCostSolver):
    def __init__(self, bcn: BooleanControlNetwork, g: Callable[[int, int], float],
                 lamb: float, theta: float,
                 Cx: Iterable[int] = None, Cu: Callable[[int], Iterable[int]] = None):
        """
        Initialize the solver.

        :type theta: threshold
        :param bcn: a Boolean control network
        :param g: stage cost function, g(x, u) --> cost
        :param Cx: state constraints, only states in Cx are allowed
        :param Cu: control constraints, Cu(i) gives the control that are allowed at i
        """
        super().__init__(bcn, g, lamb, Cx, Cu)
        assert theta >= 0
        self._theta = theta
        self._stage_cost: Dict[Tuple[int, int], float] = {}
        self._admissible_control: Dict[int, List[int]] = {}

    def _get_stage_cost(self, x: int, u: int) -> float:
        # compute and store the stage cost to avoid repetitive computations
        if (x, u) not in self._stage_cost:
            c = self.g(x, u)
            self._stage_cost[(x, u)] = c
        return self._stage_cost[(x, u)]

    def _get_next_state(self, x: int, u: int) -> int:
        blk = self.L[(u - 1) * self.N: u * self.N]
        return blk[x - 1]

    def _get_admissible_control(self, x) -> List[int]:
        if x not in self._admissible_control:
            control = []
            for u in range(1, self.M + 1):
                if self._is_input_allowed(x, u) and self._get_next_state(x, u) in self.Cx:
                    control.append(u)
            assert control
            self._admissible_control[x] = control
        return self._admissible_control[x]

    def solve(self, x0: int) -> Tuple[float, Dict[int, int]]:
        # value function
        V = {x: 0 for x in self.Cx}
        n_iter = 0
        while True:
            n_iter += 1
            psi = 0
            for x in self.Cx:
                v = V[x]
                V[x] = min(self._get_stage_cost(x, u) + self.lamb * V[self._get_next_state(x, u)]
                           for u in self._get_admissible_control(x))
                psi = max(psi, abs(v - V[x]))
            if psi < self._theta:
                print('#iterations in value iteration: ', n_iter)
                break
        # optimal policy
        pi_star = {}
        for x in self.Cx:
            v_star, u_star = min(((self._get_stage_cost(x, u) + self.lamb * V[self._get_next_state(x, u)], u)
                                 for u in self._get_admissible_control(x)), key=itemgetter(0))
            pi_star[x] = (u_star, v_star)
        # state feedback
        K = {}
        for i, (u, _) in pi_star.items():
            K[i] = u
        return pi_star[x0][1], K

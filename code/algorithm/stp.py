"""
Semi-tensor product.

This is an (incomplete) Python & numpy implementation of Dr. Cheng Daizhan's STP toolbox (http://lsc.amss.ac.cn/~dcheng/).
We try to follow the implementation details of that toolbox as much as possible.
Please refer to the documentation of that toolbox for more details.
"""
import numpy as np
from typing import Iterable

def _left_sp(A, B):
    m, n = A.shape
    p, q = B.shape
    k = n // p
    C = np.empty((m, k * q), dtype=np.result_type(A, B))
    for i in range(m):
        for j in range(q):
            C[i, j * k : (j + 1) * k] = B[:, j].reshape((1, p)) @ A[i].reshape((p, k))
    return C


def sp(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Semi-Tensor Product of Matrices using Kronecker product.
    This function combines `sp` and `sp1` of Cheng's STP toolbox.

    Note: if either A or B is a Boolean matrix, then the other will be converted to a Boolean matrix automatically.
    In such a case, the STP will be a Boolean one, i.e., AND for multiplication and OR for addition.

    Time complexity:
        Suppose A is mxn and B is pxq matrix.
    """
    assert A.ndim == 2, 'Only 2d array (matrix) is allowed'
    assert B.ndim == 2, 'Only 2d array (matrix) is allowed'
    m, n = A.shape
    p, q = B.shape
    if np.issubdtype(A.dtype, np.bool_) or np.issubdtype(B.dtype, np.bool_):
        A = A.astype(np.bool_, copy=False)
        B = B.astype(np.bool_, copy=False)
    if n == p:
        return A @ B

    # special matrices: to speed up
    if n % p == 0:
        return _left_sp(A, B)
    if p % n == 0:
        return _left_sp(B.T, A.T).T
    # general matrices
    z = np.lcm(n, p)
    d = np.result_type(A, B)
    return np.kron(A, np.eye(z // n, dtype=d)) @ np.kron(B, np.eye(z // p, dtype=d))


def logical_matrix_from(L: Iterable, n: int, dtype=np.int8) -> np.ndarray:
    """
    Reconstruct a 2d logical matrix from a 1d representation.
    An item `i` in `Lm` represents `\delta_n^i`.
    The type of the matrix is specified by `dtype`, which can be or a sub-type of `np.number` or `np.bool_`
    """
    m = np.full((n, len(L)), False, dtype=dtype)
    one = True if np.issubdtype(dtype, np.bool_) else 1
    for j, i in enumerate(L):
        m[i - 1, j] = one
    return m

def swap_matrix(m: int, n: int) -> np.ndarray:
    """
    Construct a swap matrix W_{[m, n]}, whose size is mn-by-mn.
    Complexity: O(m^2n^2)
    :param m:  int
    :param n: int
    :return: a swap matrix (a Boolean/logical matrix)
    """
    assert m > 0 and n > 0
    W = np.zeros((m * n, m * n), dtype=np.bool_)
    for i in range(m):
        for j in range(n):
            c = i * n + j
            r = j * m + i
            W[r, c] = 1
    return W


if __name__ == '__main__':
    X = np.array([1, 2, 3, -1]).reshape((1, 4))
    Y = np.array([1, 2]).reshape((2, 1))
    Z = sp(X, Y)
    print(Z)
    X = np.array([[1, 2, 1, 1], [2, 3, 1, 2], [3, 2, 1, 0]])
    Y = np.array([[1, -2], [2, -1]])
    print(sp(X, Y))
    A = np.array([[1], [2]])
    B = np.array([[2, 1], [3, 5]])
    print(sp(A, B))
    C = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.bool_)
    E = np.array([[0, 1], [0, 0], [1, 1]], dtype=np.bool_)
    D = np.array([[True], [False]])
    print(sp(C, D))
    print(sp(C, E))
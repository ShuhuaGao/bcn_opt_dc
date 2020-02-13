"""
Boolean matrix and Boolean algebra.
"""
from typing import Iterable
import numpy as np
from . import stp


def to_boolean(a: Iterable) -> np.ndarray:
    """
    Change a iterable `a` to a Boolean array.
    If 'a' is already a Boolean array, then it is returned directly.
    """
    return np.array(a, dtype='bool', copy=False)


def product(a: Iterable, b: Iterable) -> np.ndarray:
    return to_boolean(a) @ to_boolean(b)


def add(a: Iterable, b: Iterable)-> np.ndarray:
    return to_boolean(a) + to_boolean(b)

def join(a: Iterable, b: Iterable)-> np.ndarray:
    """
    Element-wise OR.
    :param a:
    :param b:
    :return:
    """
    return np.logical_or(a, b)

def meet(a: Iterable, b: Iterable) -> np.ndarray:
    """
    Element-wise AND
    """
    return np.logical_and(a, b)

def sp(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Semi-tensor product. A Boolean matrix is returned.
    """
    assert A.ndim == 2, 'Only 2d array (matrix) is allowed'
    assert B.ndim == 2, 'Only 2d array (matrix) is allowed'
    m, n = A.shape
    p, q = B.shape
    A = to_boolean(A)
    B = to_boolean(B)
    if n == p: # degrade to normal matrix product
        return product(A, B)
    return stp.sp(A, B)

def row_sum(A: np.ndarray) -> np.ndarray:
    """
    Boolean summations of rows.

    :param A: a 2d array
    :return: a 1d Boolean array
    """
    return np.sum(to_boolean(A), axis=1, dtype=np.bool_)

def col_sum(A: np.ndarray) -> np.ndarray:
    """
    Boolean summations of columns.

    :param A: a 2d array
    :return: a 1d Boolean array
    """
    return np.sum(to_boolean(A), axis=0, dtype=np.bool_)





"""
Some utility functions

"""
from typing import Tuple, List
import networkx as nx


def read_network(file: str) -> Tuple[int, int, List[int]]:
    """
    Read a Boolean network from a text file:
        Line 1: number of state variables
        Line 2: number of control inputs
        Line 3: transition matrix of the network (linear representation of a logical matrix)

    :param file: a text file
    :return: (n, m, Lm), where
        n: number of state variables
        m: number of control inputs
        Lm: network transition matrix
    """
    with open(file, 'r') as f:
        n = int(f.readline().strip())
        m = int(f.readline().strip())
        N = 2 ** n
        M = 2 ** m
        line = f.readline().strip()
        assert line, f'network transition matrix must be provided!'
        numbers = line.split()
        assert len(numbers) == M * N, f'The transition matrix must have {M * N} columns'
        L = [int(num) for num in numbers]
        for i in L:
            assert 1 <= i <= N, f'All integers in the network transition matrix must be in range [1, {N}]'
        return n, m, L


def visualize(g: nx.DiGraph, to_file: str, layout: str='fdp', **kwargs):
    from networkx.drawing.nx_agraph import to_agraph
    import pygraphviz
    g.graph['edge'] = {'arrowsize': '0.4', 'splines': 'curved'}
    g.graph['node'] = {'shape': 'circle', 'fontsize': '10', 'width': '0.2', 'margin': 0}
    g.graph['graph'] = {'scale': 1, 'nodesep': 0.1, 'ranksep': 0.1, 'margin': 0}
    ag: pygraphviz.agraph.AGraph = to_agraph(g)
    # ag.node_attr['nodesep'] = 0.1
    ag.layout(layout)
    ag.draw(to_file)


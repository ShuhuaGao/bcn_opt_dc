"""
Given a Boolean function/network, get its algebraic state-space representation.

A logical vector `\delta_n^i` is represented by an integer `i` for space efficiency. Consequently, a logical matrix
is represented by a list, each element for one column, (also known as the "condensed form").

[1] Conversion from an infix expression to a postfix one:
https://runestone.academy/runestone/books/published/pythonds/BasicDS/InfixPrefixandPostfixExpressions.html

[2] Logical connectives: https://en.wikipedia.org/wiki/Logical_connective

Author: Gao Shuhua
"""
import operator
import os
from typing import List, Union, Tuple, Iterable, Dict
from .bcn import BooleanNetwork, BooleanControlNetwork


_COMMENT = '#'
_STATES = '[STATES]'
_CONTROLS = '[CONTROLS]'

class LogicalConnective:
    """
    Represent a logical connective. https://en.wikipedia.org/wiki/Logical_connective
    """
    def __init__(self, id: str, description: str, arity: int, precedence: int, function):
        """
        Initialize a logical connective.

        :param id: a unique description
        :param description: a description text
        :param arity: number of operands
        :param precedence: operator precedence
        :param function: callable, the underlying operation which accepts *arity* argments
        """
        self.id = id
        self.description = description
        self.arity = arity
        self.precedence = precedence  # a smaller number means a higher precedence
        self.function = function

    def __str__(self):
        return self.id

    def __call__(self, *args):
        return self.function(*args)


def _imply(a, b):
    if a:
        return b
    return 1


def _xnor(a, b):
    return a == b


LOGICAL_CONNECTIVES = {
    'NOT': LogicalConnective('NOT', 'not', 1, 0, operator.not_),
    'XOR': LogicalConnective('XOR', 'exclusive disjunction', 2, 1, operator.xor),
    'AND': LogicalConnective('AND', 'and', 2, 2, operator.and_),
    'OR': LogicalConnective('OR', 'or', 2, 3, operator.or_),
    'IMPLY': LogicalConnective('IMPLY', 'implication', 2, 4, _imply),
    'EQUIV': LogicalConnective('EQUIV', 'equivalent', 2, 5, _xnor)
}


def _infix_to_postfix(expression: str) -> List[Union[LogicalConnective, str]]:
    """
    Convert an infix expression to its postfix form.

    :param expression: infix, separated by spaces
    :return: postfix expression, a list, whose element is an operator (LogicalConnective) or a variable (str)
    """
    # parse tokens: handle ( and ) specially, which may not be separated by spaces, e.g., 'A OR (B AND C)'
    items = expression.split()
    tokens = []
    for item in items:
        token = ''
        for c in item:
            if c in '()':
                if token:
                    tokens.append(token)
                    token = ''
                tokens.append(c)
            else:
                token = token + c
        if token:
            tokens.append(token)
    # conversion
    op_stack = []
    output = []
    for token in tokens:
        if token.upper() in LOGICAL_CONNECTIVES:   # an operator
            connective = LOGICAL_CONNECTIVES[token.upper()]
            while op_stack and isinstance(op_stack[-1], LogicalConnective) and \
                    op_stack[-1].precedence < connective.precedence:
                output.append(op_stack.pop())
            op_stack.append(connective)
        elif token == '(':
            op_stack.append(token)
        elif token == ')':
            left_parenthesis_found = False
            while op_stack:
                top = op_stack.pop()
                if top == '(':
                    left_parenthesis_found = True
                    break
                else:
                    output.append(top)
            if not left_parenthesis_found:
                raise RuntimeError("Unmatched parentheses are encountered: an extra ')'!")
        elif token.upper() in ['1', 'TRUE']:
            output.append('TRUE')
        elif token.upper() in ['0', 'FALSE']:
            output.append('FALSE')
        else:  # a variable
            output.append(token)
    while op_stack:
        top = op_stack.pop()
        if top == '(':
            raise RuntimeError("Unmatched parentheses are encountered: an extra '('!")
        output.append(top)
    return output


def _evaluate_postfix(expression, values: {}):
    """
    Evaluate a postfix expression with the given parameter values.

    :param expression: postfix
    :param values: a dict: variable --> value (0/1 or False/True)
    :return: a Boolean variable, or 0/1
    """
    operand_stack = []
    for token in expression:
        if isinstance(token, str):  # a variable
            if token in values:
                val = values[token]
                operand_stack.append(val)
            elif token == 'TRUE':
                operand_stack.append(True)
            elif token == 'FALSE':
                operand_stack.append(False)
            else:
                raise RuntimeError(f"Unrecognized variable: '{token}'")
        else:   # a logical connective
            arguments = []
            for _ in range(token.arity):
                arguments.append(operand_stack.pop())
            result = token(*arguments[::-1])
            operand_stack.append(result)
    return operand_stack.pop()


def _assr_function(pf_expr: List[Union[LogicalConnective, str]], states: List[str], controls: List[str]) -> List[int]:
    """
    Compute the ASSR for a Boolean function.

    :param pf_expr:  the postfix expression of a Boolean function
    :param states: the state variables
    :param controls: the control inputs. If `None`, then no inputs.
    :return: the structure matrix, a list of length MN
    """
    n = len(states)
    m = len(controls)
    N = 2 ** n
    M = 2 ** m
    MN = M * N
    all_variables = controls + states
    structure_matrix = [None] * MN
    # enumerate the binary sequences to get the truth table
    for h in range(MN):
        bh = f'{h:0{m+n}b}'
        values = {var: int(val) for var, val in zip(all_variables, bh)}
        output =  _evaluate_postfix(pf_expr, values)
        k = MN - h
        if output: # 1 (True)
            structure_matrix[k - 1] = 1
        else:
            structure_matrix[k - 1] = 2
    return structure_matrix


def _tokenize(state_to_expr: Dict[str, str],  controls: Iterable[str]=None) -> Tuple[Dict[str, List[Union[LogicalConnective, str]]], List[str]]:
    """
    (1) Parse the `exprs` into postfix forms
    (2) Infer the control inputs, if `controls` is `None`

    :return: the tokenized expressions and the controls
    """
    state_to_pf_expr = {s: _infix_to_postfix(e) for s, e in state_to_expr.items()}
    if controls is None:
        # infer controls
        controls = []
        for pf_expr in state_to_pf_expr.values():
            for t in pf_expr:
                if isinstance(t, str): # t is a variable, or 'TRUE' or 'FALSE'
                    if t not in ['TRUE', 'FALSE'] and t not in state_to_pf_expr:    # a control
                        if t not in controls:
                            controls.append(t)
    else:
        controls = list(controls)
    # validate
    for s, pf_expr in state_to_pf_expr.items():
        for t in pf_expr:
            if isinstance(t, str):
                assert t in state_to_pf_expr or t in controls, f"Unrecognized variable: '{t}' in equation of {s}"
    return state_to_pf_expr, controls


def _assr_network(state_to_pf_expr: Dict[str, List[Union[LogicalConnective, str]]], states: List[str],
                  controls: List[str], verbose: bool=True) -> List[int]:
    """
    Get the ASSR of a Boolean (control) network.

    :param state_to_pf_expr: state -> its postfix expression
    :param states: state variables
    :param controls: control inputs.
    :return: network transition matrix, each column is represented by an integer
    """
    assert len(state_to_pf_expr) == len(states), 'The number of Boolean functions must be equal to the number of state states'
    # get the structure matrix of each state (i.e., its Boolean equation)
    state_to_sms = {}
    for s, pf_expr in state_to_pf_expr.items():
        if verbose:
            print(f'\tComputing the structure matrix for state {s} ...')
        state_to_sms[s] = _assr_function(pf_expr, states, controls)
    n = len(states)
    m = len(controls)
    transition_matrix = [None] * (2 ** m * 2 ** n)
    stp = lambda i, j: (i - 1) * 2 + j
    if verbose:
        print('\tComposing the complete network transition matrix...')
    for k in range(len(transition_matrix)):  # k-th column
        r = 1
        for s in states:
            sm = state_to_sms[s]
            r = stp(r, sm[k])
        transition_matrix[k] = r
    return transition_matrix

def build_ASSR(source: Union[str, Iterable[str]], states: List[str]=None,
               controls: List[str]=None, verbose: bool=True) -> Union[BooleanNetwork, BooleanControlNetwork]:
    """
    Build the ASSR for a given Boolean network in a string form.
    Each Boolean function is given by the form: state = f(states, controls).
    If a text file is given, each Boolean function is provided per line, and '#' starts a comment line

    :param source: str or a list of str. (1) str: a single Boolean function or a text file, which contains one or more
        Boolean functions (i.e., a network), each per line; (2) a list of str: multiple Boolean functions
    :param states: state variables. If `None`, then inferred automatically.
    :param controls: control inputs. If this a Boolean network with no inputs, then give it an empty List.
        If `None`, then inferred automatically.
    :param verbose: whether to print more information
    :return: a Boolean network if there are no inputs; otherwise, a Boolean control network

    .. note::
        If the states and controls are inferred, the order of states corresponds to the line order, whereas the order
        of controls depend on their appearance order in the equations. To precisely control the order (especially for
        controls), two additional lines may be appended after the state equations that begin with "[STATES]" or "[CONTROLS]".
        For example, line "[STATES] AKT MKK EGFR" specifies the state order (AKT, MKK, EGFR).
        Of course, both "[STATES]" and "[CONTROLS]" lines are optional.
        The non-None arguments `states` and `controls` have higher precedence than "[STATES]" and "[CONTROLS]" lines respectively.
    """
    # get the strings of a network
    net = []
    if isinstance(source, str):
        if os.path.isfile(source):
            if verbose:
                print(f'User provided a network file: {source}\nParsing...')
            with open(source, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(_COMMENT):
                        continue
                    elif line.startswith(_STATES):
                        if states is None:
                            words = line.split()
                            states = [w.strip() for w in words[1:]]
                    elif line.startswith(_CONTROLS):
                        if controls is None:
                            words = line.split()
                            controls = [w.strip() for w in words[1:]]
                    else:
                        if line:  # skip empty lines if any
                            net.append(line)
        else:
            if verbose:
                print(f'User provided a single Boolean equation.')
            net.append(source)
    else:
        if verbose:
            print(f'User provided a list of Boolean equations.')
        net = list(source)
    # extract the states and equations
    state_to_expr = {}
    inferred_states = []
    for eq in net:
        state, expr = eq.split('=')
        state = state.strip()
        expr = expr.strip()
        if states is not None:
            assert state in states, f'Unexpected state {state} is encountered!'
        else:
            inferred_states.append(state)
        assert state not in state_to_expr, f'More than one equation is provided for state {state}'
        state_to_expr[state] = expr
    if states is not None:
        for s in states:
            assert s in state_to_expr, f'The equation for state {s} is missing'
    else:
        states = inferred_states
    if verbose:
        print('Tokenizing...')
    # tokenize
    state_to_pf_expr, controls = _tokenize(state_to_expr, controls)
    assert set(states).isdisjoint(controls), 'States and controls should be disjoint'
    if verbose:
        print(f'States are {states}')
        print(f'Controls are {controls}')
        print('Computing...')
    # get the ASSR the network
    L = _assr_network(state_to_pf_expr, states, controls, verbose)
    # wrap them into a Boolean (control) network
    m = len(controls)
    n = len(states)
    if m == 0:
        return BooleanNetwork(n, L, states)
    return BooleanControlNetwork(n, m, L, states, controls)

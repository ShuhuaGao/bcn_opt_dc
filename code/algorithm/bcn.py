"""
Boolean (control) network.
"""
from typing import List


class BooleanNetwork:
    def __init__(self, n: int, L: List[int], state_variable_names: List[str]=None):
        assert n > 0
        self.n = n
        self.N = 2 ** n
        self._assert_L_length(L)
        self.L = L
        self.states = state_variable_names

    def _assert_L_length(self, L):
        assert len(L) == self.N, 'The length of L is invalid'

    def to_file(self, file: str, write_states: bool=False):
        """
        Write this network into a text file in the following format.
        Line 1: n
        Line 2: m
        Line 3: L, each item separated by space
        Line 4 (optional): states, each item separated by space. This line begins with '[STATES] '.

        :param file: a text file to be written
        :param write_states: whether write the state names into the file
        """
        with open(file, 'w') as f:
            f.write(f'{self.n}\n')
            f.write(f'{self.m}\n')
            f.write(' '.join(str(i) for i in self.L))
            f.write('\n')
            if write_states:
                f.write('[STATES] ')
                f.write(' '.join(self.states))

    def __str__(self):
        return f'n = {self.n}\nL = {self.L}'


class BooleanControlNetwork(BooleanNetwork):
    def __init__(self, n: int, m: int, L: List[int], state_variable_names: List[str]=None, control_variable_names: List[str]=None):
        assert m > 0
        self.m = m
        self.M = 2 ** m
        super().__init__(n, L, state_variable_names)
        self.controls = control_variable_names

    def _assert_L_length(self, L):
        assert len(L) == self.M * self.N, 'The length of L is invalid'

    def to_file(self, file: str, write_states_controls: bool = False):
        """
        Write this network into a text file in the following format.
        Line 1: n
        Line 2: m
        Line 3: L, each item separated by space
        Line 4 (optional): states, each item separated by space. This line begins with '[STATES] '
        Line 5 (optional): controls, each item separated by space. This line begins with '[CONTROLS] '

        :param file: a text file to be written
        :param write_states_controls: whether write the state names into the file
        """
        super().to_file(file, write_states_controls)
        if write_states_controls: # append the controls line
            with open(file, 'a') as f:
                f.write('\n')
                f.write('[CONTROLS] ')
                f.write(' '.join(self.controls))

    def __str__(self):
        return f'n = {self.n}, m = {self.m}\nL = {self.L}'



# This module has been developed by Gabriel Fonseca and Steve Furber at The University of Manchester as part of the
# paper:
#
# "Using Stochastic Spiking Neural Networks on SpiNNaker to Solve Constraint Satisfaction Problems"
# Submitted to the journal Frontiers in Neuroscience| Neuromorphic Engineering
# -----------------------------------------------------------------------------------------------------------------------
"""Class that translates Sudoku puzzles to constraint satisfaction problems.

This module contains the sudoku2csp class which accept puzzles formated as a python 9x9 array of the form:

[[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0]]

in the previous grid there are no clues, you can change the zeros for appropriate clues.
"""


class sudoku2csp:
    """Class that translates Sudoku puzzles to constraint satisfaction problems.

    From the 9x9 arrat CSP variables are defined enumerating from leftmost digit of the first (top) row to the rightmost
    of the last (bottom) row. Constraints are defined in a directed way so the snn_creator should build the other half
    from symmetry arguments.
    """

    def __init__(self, clue_list=[]):
        """Initialize the sudoku2csp class.

        args:
            clue_list: a python 9x9 array representing the Sudoku puzzle, blank cells are given with the digit 0.
        """
        # Define CSP variables for Sudoku,
        variables = []
        size = len(clue_list)
        sub_size = int(size ** (1.0 / 2.0))
        for i in range(size):
            for j in range(size):
                variables.append([i, j])
        # Define CSP constraints for Sudoku.
        constraints = []
        # Horizontal and vertical constraints.
        for var1, xy1 in enumerate(variables):
            for var2, xy2 in enumerate(variables):
                if (xy2[0] == xy1[0] or xy2[1] == xy1[1]) and var2 > var1:
                    constraints.append([var1, var2])
        # 3x3 squares diagonal constraints.
        for var1, xy1 in enumerate(variables):
            for var2, xy2 in enumerate(variables):
                # below:  same 3X3 square & (different row & different column) & var2>var1
                if (
                    (
                        xy2[0] // 3 == xy1[0] // 3
                        and xy2[1] // sub_size == xy1[1] // sub_size
                    )
                    and (xy2[0] != xy1[0] and xy2[1] != xy1[1])
                    and var2 > var1
                ):
                    constraints.append([var1, var2])
        # Apply format to constraints.
        formatted_constraints = []
        for constraint in constraints:
            formatted_constraints.append(
                {"source": constraint[0], "target": constraint[1]}
            )
        # Format clues for set_clues method in CSP class.
        if clue_list:
            # flatten puzzle input
            digits = []
            for i in clue_list:
                for k in i:
                    digits.append(k)
            # Create list where the first array are the CSP variables and the second their values, to be used when
            # calling the method set_clues of the CSP class.
            clues = [[], []]
            for i, clue in enumerate(digits):
                if clue != 0:
                    clues[0].append(i)
                    clues[1].append(clue - 1)
        self.variables = variables
        self.constraints = formatted_constraints
        self.var_num = len(variables)
        self.cons_num = len(constraints)
        self.dom_num = size
        self.clues = clues

    def var_grid(self):
        """Print the CSP variables in a 9X9 square format"""
        n = 0
        for i in range(9):
            print(self.variables[n : n + 9])
            n += 9
        print("\n Total number of variables: %d" % (len(self.variables)))
        print("\n Total number of constraints: %d" % (len(self.constraints)))

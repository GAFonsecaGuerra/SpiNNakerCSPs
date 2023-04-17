# This module has been developed by Gabriel Fonseca and Steve Furber at The University of Manchester as part of the
# paper:
#
# "Using Stochastic Spiking Neural Networks on SpiNNaker to Solve Constraint Satisfaction Problems"
# Submitted to the journal Frontiers in Neuroscience| Neuromorphic Engineering
# -----------------------------------------------------------------------------------------------------------------------
"""Class that defines a constraint satisfaction problem representing a lattice of Ising spins.

This module contains the spin2csp class which allows the definition of 1-dimensional, 2-dimensional or 3-dimensional
regular lattices where nodes represent Ising spins which interact ferromagnetically or antiferromagnetically with their
nearest-neighbours. The problem is described as a constraint satisfaction problem in which variables are numbered
"""
import numpy as np


class spin2csp:
    def __init__(self, length, dimensions=3):
        chain = range(length**dimensions)
        lattice = np.reshape(chain, tuple([length] * dimensions))
        constraints = []
        if dimensions == 3:
            for x in range(length):
                for y in range(length):
                    for z in range(length):
                        if x != length - 1:
                            constraints.append((lattice[x][y][z], lattice[x + 1][y][z]))
                        if y != length - 1:
                            constraints.append((lattice[x][y][z], lattice[x][y + 1][z]))
                        if z != length - 1:
                            constraints.append((lattice[x][y][z], lattice[x][y][z + 1]))
                        if x >= 0:
                            constraints.append((lattice[x][y][z], lattice[x - 1][y][z]))
                        if y >= 0:
                            constraints.append((lattice[x][y][z], lattice[x][y - 1][z]))
                        if z >= 0:
                            constraints.append((lattice[x][y][z], lattice[x][y][z - 1]))
                    # Create periodic condition on boundaries:
                    constraints.append((lattice[x][y][length - 1], lattice[x][y][0]))
                    constraints.append((lattice[x][length - 1][y], lattice[x][0][y]))
                    constraints.append((lattice[length - 1][x][y], lattice[0][x][y]))
        elif dimensions == 2:
            for x in range(length):
                for y in range(length):
                    if x != length - 1:
                        constraints.append((lattice[x][y], lattice[x + 1][y]))
                    if x >= 0:
                        constraints.append((lattice[x][y], lattice[x - 1][y]))
                    if y != length - 1:
                        constraints.append((lattice[x][y], lattice[x][y + 1]))
                    if y >= 0:
                        constraints.append((lattice[x][y], lattice[x][y - 1]))
                # Create periodic condition on boundaries:
                constraints.append((lattice[x][length - 1], lattice[x][0]))
                constraints.append((lattice[length - 1][x], lattice[0][x]))
        else:  # dimensions == 1:
            for x in range(length):
                if x != length - 1:
                    constraints.append((lattice[x], lattice[x + 1]))
                if x >= 0:
                    constraints.append((lattice[x], lattice[x - 1]))
            # Create periodic condition on boundaries:
            constraints.append((lattice[length - 1], lattice[0]))
        formatted_constraints = []
        for constraint in constraints:
            formatted_constraints.append(
                {"source": constraint[0], "target": constraint[1]}
            )

        self.array = lattice
        self.var_num = length**dimensions
        self.constraints = formatted_constraints
        self.cons_num = len(constraints)  # it should equal (length**3)*6-(length**2)*6
        self.dom_num = 2

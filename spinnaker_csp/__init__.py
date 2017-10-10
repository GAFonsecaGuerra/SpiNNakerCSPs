"""
  Spiking Neural Network Solver for Constraint Satisfaction Problems.

This package has been developed by Gabriel Fonseca and Steve Furber at The University of Manchester as part of the 
paper:

"Using Stochastic Spiking Neural Networks on SpiNNaker to Solve Constraint Satisfaction Problems"
Submitted to the journal Frontiers in Neuroscience| Neuromorphic Engineering

A constraint satisfaction problem (CSP) is generally described by a set of variables {X_i} taking values over a certain 
set of domains {D_i}, over which the set of constraints {C_ij} is imposed. The problem consists in finding assignations
for all variables X_i's so that all C_ij's are satisfied.

To solve a particular CSP import the CSP class from snn_solver.py and create an instance object. To build the solver network
you may implement methods of the CSP class 

Additionally, you will be able to keep track of the network dynamics through the functions available in the analysis.py module
e.g. an entropy vs. time plot which colourizes distinctively the regions satisfying the constraints (in Blue) and the ones violating 
them (in Red). If an unsatisfying configuration is visited again it will be plotted in green. 

Depending on the CSP, you will have a straightforward or a complicated description of the constraints, for the later
it is advised to create a script translating the original problem to a CSP description, examples of this procedure
have been included for:

Sudoku:                      sudoku2csp.py     
Ising Spin Systems           spin2csp.py
Travelling Salesman Problem: tsp2csp.py                

when the constraints are simple to describe, e.g. for an antiferromagnetic spin chain, we generate the constraints list
on the example script. In the examples subdirectory of the project folder (SpiNNakerCSPs) we provide example implementations for:

Coloring Map Problem:        australia_cmp.py and world_cmp.py
Sudoku Puzzle:               sudoku_easy.py, sudoku_hard.py and escargot.py
Spin Systems:                spin_system.py

In the case of Sudoku, you will find a sudoku_puzzles.py inside the puzzles folder containing some example puzzles to be 
imported on examples/sudoku.py
"""

from snn_creator import CSP
from analysis import plot_entropy
from translators.sudoku2csp import sudoku2csp
from translators.spin2csp import spin2csp
from translators.world_bordering_countries import world_borders, world_countries
from puzzles.sudoku_puzzles import puzzles

__all__ = ['snn_creator', 'analysis','CSP','plot_entropy','sudoku2csp','puzzles','spin2csp','world_borders',
           'world_countries']

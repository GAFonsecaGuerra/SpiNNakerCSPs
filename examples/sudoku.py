# This module has been developed by Gabriel Fonseca and Steve Furber at The University of Manchester as part of the
# paper:
#
# "Using Stochastic Spiking Neural Networks on SpiNNaker to Solve Constraint Satisfaction Problems"
# Submitted to the journal Frontiers in Neuroscience| Neuromorphic Engineering
#-----------------------------------------------------------------------------------------------------------------------
"""
Spiking Neural Network Solver for Sudoku puzzles.

Some example puzzles are available to be imported from the puzzles folder.
"""
import spynnaker7.pyNN as p                                           # simulator
import argparse

try:
    from spinnaker_csp import CSP, plot_entropy, sudoku2csp, puzzles
except:
    import sys
    import os
    sys.path.append(os.getcwd()+'/..')
    from spinnaker_csp import CSP, plot_entropy, sudoku2csp, puzzles

#Take puzzle as argument
parser = argparse.ArgumentParser(description='''This script creates a spiking neural network representation of a
                                             Sudoku puzzle whose dynamics implements a stochastic search for
                                             satisfiability.''')
parser.add_argument("puzzle", help='name of puzzle in puzzles dictionary: easy, hard, AI_escargot or platinum_blonde',
                    type=str, default='easy')
args = parser.parse_args()

name=args.puzzle
grid=puzzles[name][1]

# Show puzzle in std output.
for i in range(9):
    print '''SpiNNaker will run the stochastic search simulation for:
        %s'''%str(grid[i])
# SpiNNaker setup.
run_time = 60000         # simulation run time
p.setup(timestep=1.0)    # SpiNNaker machine setup
#translate puzzle to CSP form
sk = sudoku2csp(grid)
# Build spiking neural network.
sudoku = CSP(sk.var_num, sk.dom_num, sk.constraints, core_size=25, run_time=run_time)       # create csp instance
sudoku.clues_inhibition = True
sudoku.set_clues(sk.clues)
sudoku.build_domains_pops()
sudoku.build_stimulation_pops(1, full=True, stim_ratio=1.)
sudoku.build_dissipation_pops()
sudoku.internal_inhibition(w_range=[-0.2/2.5, 0.0])
sudoku.stimulate_cores(w_range=[1.4, 1.6], d_range=[1.0, 1.0], w_clues=[1.8, 2.0]) # , w_clues=[1.5, 1.9])
sudoku.apply_constraints(w_range=[-0.2/2.5, 0.0])
sudoku.initialize()
# Record spikes from variable populations.
sudoku.recording()
# Run simulation.
p.run(run_time)
# Save recorded spikes.
sudoku.save(name, False)
sudoku.report_network_params('report_sudoku_%s'%name)
# End simulation.
p.end()
# Plot entropy.
sol = plot_entropy(name, 200)
# Show solution on std output.
if sol is not None:
    for i in range(len(sol)): sol[i]=sol[i]+1
    print '='*70
    n=0
    for i in range(9):
        print sol[n:n+9]
        n+=9

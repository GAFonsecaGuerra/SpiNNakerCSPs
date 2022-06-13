# This module has been developed by Gabriel Fonseca and Steve Furber at The University of Manchester as part of the
# paper:
#
# "Using Stochastic Spiking Neural Networks on SpiNNaker to Solve Constraint Satisfaction Problems"
# Submitted to the journal Frontiers in Neuroscience| Neuromorphic Engineering
# -----------------------------------------------------------------------------------------------------------------------
"""
Coloring map problem for the map of australia.

The map and constraints are from the book:

Artificial Intelligence: A Modern Approach, Second Edition) by Stuart Russell and Peter Norvig
chapter 5, figure 5.1, accessed from http://aima.cs.berkeley.edu/2nd-ed/newchap05.pdf

territories are labeled as follows:
1 Western Australia
2 Northern Territory
3 Queensland
4 New South Wales
5 Victoria
6 South Australia
0 Tasmania

The coloring uses 3 colors red -> 0, green->1, blue->2
"""
import spynnaker8 as p  # simulator
import argparse

try:
    from spinnaker_csp import CSP, plot_entropy
except:
    import sys
    import os

    sys.path.append(os.getcwd() + "/..")
    from spinnaker_csp import CSP, plot_entropy

parser = argparse.ArgumentParser(
    description="""This script creates a spiking neural network representation of the
                                              problem of 3-colouring the map of australia whose dynamics implements a
                                              stochastic search for satisfiability."""
)
parser.add_argument("--i_offset", help="offset current", type=float, default=0.7)
parser.add_argument(
    "--phase", help="phase shift for depression", type=float, default=100
)
parser.add_argument(
    "--w_inh", help="inhibition weight for depression", type=float, default=-1.5
)
parser.add_argument(
    "-n", "--name", help="name to use for naming files", type=str, default="world_cmp"
)
args = parser.parse_args()

name = args.name

bordering_states = [
    {"source": 1, "target": 2},
    {"source": 1, "target": 6},
    {"source": 2, "target": 3},
    {"source": 2, "target": 6},
    {"source": 3, "target": 4},
    {"source": 3, "target": 6},
    {"source": 4, "target": 5},
    {"source": 4, "target": 6},
    {"source": 5, "target": 6},
]

# SpiNNaker setup.
run_time = 30000
p.setup(timestep=1.0)
# Build spiking neural network.
map = CSP(variables_number=7, domain_size=3, constraints=bordering_states, core_size=10)
map.cell_params_lif["i_offset"] = 0.2
map.clues_inhibition = False
map.build_domains_pops()
map.build_stimulation_pops(1, stim_ratio=1, full=True)
map.build_dissipation_pops()
map.internal_inhibition(w_range=[-1.2, -1.5], d_range=[1.0, 1.0])
map.stimulate_cores(w_range=[0.05, 0.3], d_range=[1.0, 1.0])
map.depress_cores(w_range=[-1.5, -2.0], d_range=[1.0, 1.0])
map.apply_constraints(w_range=[1.2, 1.4], d_range=[1.0, 1.2])
# Record spikes from variable populations.
map.recording()
# Run simulation.
p.run(run_time)
# Save recorded spikes.
map.save("australia")
map.report_network_params("report_cmp_australia")
# End simulation.
p.end()
# Plot entropy.
sol = plot_entropy("australia", 200, preprint=False)
formatted = []
for i in sol:
    formatted.append("{{%d}}" % (i + 1))
print("".join(formatted))

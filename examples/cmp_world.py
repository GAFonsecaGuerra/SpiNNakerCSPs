# This module has been developed by Gabriel Fonseca and Steve Furber at The University of Manchester as part of the
# paper:
#
# "Using Stochastic Spiking Neural Networks on SpiNNaker to Solve Constraint Satisfaction Problems"
# Submitted to the journal Frontiers in Neuroscience| Neuromorphic Engineering
#-----------------------------------------------------------------------------------------------------------------------
"""
Spiking Neural Network Solver for the problem of 4-colouring the map of the world.

This module corresponds to the coloring map problem for the map of the world including 193 countries which are
represented by numbers according to the list in alphabetical order as listed in the 
 CSP.world_countries dictionary. 
the constraints are listed in textual_border_countries.py in human readeble form and in world_border_countries.py for the script
the data were obtained from the mathematica notebook ColoredWorld.nb

 as seen in the CSP.world_countries 
dictionary. Constraints corresponding to bordering countries of the world as defined by the United Nations as shown in 
the CSP.world_borders dictionary. 

The countries and bordering correspond to data from the United Nations available in Mathematica Wolfram
(Wolfram Research, 2017).
"""
import spynnaker7.pyNN as p                                           # simulator
import argparse
try:
    from spinnaker_csp import CSP, plot_entropy, world_borders
except:
    import sys
    import os
    sys.path.append(os.getcwd()+'/..')
    from spinnaker_csp import CSP, plot_entropy, world_borders

parser = argparse.ArgumentParser(description='''This script creates a spiking neural network representation of the 
                                              problem of 4-colouring the map of the world whose dynamics implements a
                                              stochastic search for satisfiability.''')
parser.add_argument("--i_offset",   help='offset current',                   type=float, default=0.7)
parser.add_argument("--phase",      help='phase shift for depression',       type=float, default=100)
parser.add_argument("--w_inh",      help='inhibition weight for depression', type=float, default=-1.5)
parser.add_argument("-n", "--name", help='name to use for naming files',     type=str,   default='world_cmp')
args = parser.parse_args()

name = args.name

# SpiNNaker setup.
run_time = 30000
p.setup(timestep=1.0)
# Build spiking neural network.
map = CSP(variables_number = 193, domain_size = 4, constraints=world_borders, core_size=25,run_time=run_time)
map.cell_params_lif['i_offset']= args.i_offset
map.cell_params_lif['tau_syn_I']=5.0
map.clues_inhibition=True
map.build_domains_pops()
map.internal_inhibition(w_range=[-0.2/2.5, 0.0])  # , w_range= [-0.5,-1.0], d_range=[2.0,2.0])
map.apply_constraints(w_range=[-0.2/2.5, 0.0])  # ,  w_range=[-0.5, -1.0], d_range=[2.0, 2.0])
map.build_stimulation_pops(n_populations=5, full=True,shrink=1.0, stim_ratio=0.10, phase=5000)  # , n_populations=5,shrink=0.750, stim_ratio=0.50, full=False) #, phase=15000)
# map.stimulate_cores(map, w_range=[0.1, 1.5], d_range=[1.0,1.0])
map.build_dissipation_pops(d_populations=1,shrink=1.0, phase=0) # , d_populations=2, shrink=0.250, stim_ratio=0.2, full=False, phase=args.phase)
map.depress_cores(w_range=[-1.510, args.w_inh], d_range=[1.0,1.0])
map.initialize() #, v_range=[-65.,-55.])
# Record spikes from variable populations.
map.recording()
# Run simulation.
p.run(run_time)
# Save recorded spikes.
map.save(name)
map.report_network_params('report_%s'%name)
# End simulation.
p.end()
# Plot entropy.
plot_entropy(name, 200, show=True)
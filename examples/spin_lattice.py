# This module has been developed by Gabriel Fonseca and Steve Furber at The University of Manchester as part of the
# paper:
#
# "Using Stochastic Spiking Neural Networks on SpiNNaker to Solve Constraint Satisfaction Problems"
# Submitted to the journal Frontiers in Neuroscience| Neuromorphic Engineering
#-----------------------------------------------------------------------------------------------------------------------
"""
Spiking Neural Network Solver for magnetic configurations of Ising spin lattices.

This system represents an ensemble of spins each one with up and down as possible states, the constraints represent 
nearest-neighbours ferromagnetic, antiferromagnetic or mixed coupling. The solver works with 1-dimensional,
2-dimensional and 3-dimensional
spin systems.
"""
import spynnaker7.pyNN as p                                           # simulator
import argparse
try:
    from spinnaker_csp import CSP, plot_entropy, spin2csp
except:
    import sys
    import os
    sys.path.append(os.getcwd()+'/..')
    from spinnaker_csp import CSP, plot_entropy, spin2csp

parser = argparse.ArgumentParser(description='''This script creates a spiking neural network representation of a
                                             regular lattice of Ising spins whose dynamics implements a stochastic 
                                             search for satisfiability. Only nearest-neighbours interaction''')
parser.add_argument("lattice", help='name of lattice from: AF3D, FM3D, SG50, SG10, ring',
                    type=str, default='ring')
parser.add_argument("-n", "--name", help='name to use for naming files', type=str, default=None)
parser.add_argument("-l", "--length", help='number of spins on each side of the lattice', type=int, default=10)
args = parser.parse_args()

# Parameters for each configuration. Format: [dimensions, random coupling type?, synapse_type, probability, label].
systems={'AF3D': [3, False, "inhibitory", 1, "AF3D"],   # Antiferromagnetic 3D lattice.
         'FM3D': [3, False, "excitatory", 0, "FM3D"],   # Ferromagnetic 3D lattice.
         'SG50': [2, True, "inhibitory", 0.5, "SGI"],   # 2D spin glass, 50% probability of antiferromagnetic coupling.
         'SG10': [2, True, "inhibitory", 0.1, "SGII"],  # 2D spin glass, 10% probability of antiferromagnetic coupling.
         'ring': [1, False, "inhibitory", 1, "chain"]}  # 1D spin system with connected ends.

experiment = systems[args.lattice]
dimensions, rn, nature, pAF, name = experiment
if args.name is not None:
    name = args.name
length = 10

# Get CSP description of specified lattice.
crystal = spin2csp(length, dimensions)

# SpiNNaker setup.
run_time = 50000
p.setup(timestep=1.0)
# Build spiking neural network.
spin = CSP(variables_number=crystal.var_num, domain_size=crystal.dom_num, constraints=crystal.constraints,
           core_size=25,run_time=run_time)
spin.cell_params_lif['i_offset']=0.0 # args.i_offset
spin.build_domains_pops()
spin.build_stimulation_pops(n_populations=1, shrink=1.0, stim_ratio=1.0, full=False, phase=0.0) #shrink = 1.0
spin.build_dissipation_pops(phase= 3000)
spin.internal_inhibition()
spin.stimulate_cores() # w_range=[0.001,0.005])  # specification of weights only for FM
spin.depress_cores(w_range=[-4.5,-14.5])  # w_range=[-4.5,-14.5]
spin.apply_constraints(nature, random_cons=rn, pAF=pAF)
# Record spikes from variable populations.
spin.recording()
# Run simulation.
p.run(run_time)
# Save recorded spikes.
spin.save(name)
spin.report_network_params('report_%s'%name)
# End simulation.
p.end()
# Plot entropy.
sol = plot_entropy(name, 200, show=False, cons_type= nature)
# Show solution on std output.
if sol:
    for i in range(len(sol)):
        if sol[i] == 0: sol[i]=-1
    print('='*70)
    if dimensions==1:
        print sol
    if dimensions == 2:
        n = 0
        print'['
        for i in range(length):
            print sol[n:n + length], ','
            n+=length
        print']'

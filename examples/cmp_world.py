# This module has been developed by Gabriel Fonseca and Steve Furber at The University of Manchester as part of the
# paper:
#
# "Using Stochastic Spiking Neural Networks on SpiNNaker to Solve Constraint Satisfaction Problems"
# Submitted to the journal Frontiers in Neuroscience| Neuromorphic Engineering
# -----------------------------------------------------------------------------------------------------------------------
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
import spynnaker8 as p  # simulator
import argparse

try:
    from spinnaker_csp import CSP, plot_entropy, world_borders
except:
    import sys
    import os

    sys.path.append(os.getcwd() + "/..")
    from spinnaker_csp import CSP, plot_entropy, world_borders

parser = argparse.ArgumentParser(
    description="""This script creates a spiking neural network representation of the
                                              problem of 4-colouring the map of the world whose dynamics implements a
                                              stochastic search for satisfiability."""
)
parser.add_argument("--tau_refrac", type=float, help="refractory period", default=2.0)
parser.add_argument("--name", type=str, help="sufix to name files", default="cmp_world")
parser.add_argument(
    "--i_offset", type=float, help="constant current stimulation", default=0.0
)
parser.add_argument(
    "--diss_shrink",
    type=float,
    help="shrinkage of the time to activate/deactivate diss populations",
    default=1.0,
)
parser.add_argument(
    "--stim_pops",
    type=int,
    help="number of noise (Poisson) stimulation populations",
    default=1,
)
parser.add_argument(
    "--internal_delay_min",
    type=float,
    help="min delay for internal inhibition synapses",
    default=2.0,
)
parser.add_argument(
    "--internal_max",
    type=float,
    help="max value for internal inhibition weights",
    default=-2.4,
)
parser.add_argument(
    "--diss_full",
    type=bool,
    help="whether the dissipation should last for the full simulation",
    default=False,
)
parser.add_argument(
    "--tau_syn_E", type=float, help="excitatory synapse time constant", default=5.0
)
parser.add_argument(
    "--stim_full",
    type=bool,
    help="whether the stimulation should last for the full simulation",
    default=False,
)
parser.add_argument(
    "--diss_max", type=float, help="max value for dissipation weights", default=-1.5
)
parser.add_argument("--cm", type=float, help="membrane capacitance", default=0.25)
parser.add_argument(
    "--internal_delay_max",
    type=float,
    help="max delay for internal inhibition synapses",
    default=2.0,
)
parser.add_argument(
    "--stim_phase",
    type=int,
    help="a delay for activation  of stimulation after the simulation begins",
    default=0,
)
parser.add_argument(
    "--diss_pops",
    type=int,
    help="number of noise (Poisson) dissipation populations",
    default=1,
)
parser.add_argument(
    "--diss_ratio",
    type=float,
    help="fraction of the dissipation interval to keep dissipation active",
    default=1.0,
)
parser.add_argument(
    "--stim_delay_min",
    type=float,
    help="min value of the synaptic delay from the stimulating poisson sources to the principal neurons",
    default=1.0,
)
parser.add_argument(
    "--stim_shrink",
    type=float,
    help="shrinkage of the time to activate/deactivate stim populations",
    default=1.0,
)
parser.add_argument(
    "--v_rest", type=float, help="rest membrane potential", default=-65.0
)
parser.add_argument(
    "--activate_stim",
    type=bool,
    help="whether stimulation with Poisson sources should be activated",
    default=True,
)
parser.add_argument(
    "--stim_delay_max",
    type=float,
    help="max value of the synaptic delay from the stimulating poisson sources to the principal neurons",
    default=1.0,
)
parser.add_argument(
    "--stim_max", type=float, help="max value for stimulation weights", default=1.5
)
parser.add_argument(
    "--internal_min",
    type=float,
    help="min value for internal inhibition weights",
    default=-2.1,
)
parser.add_argument(
    "--stim_ratio",
    type=float,
    help="fraction of the stimulation interval to keep stimulation active",
    default=1.0,
)
parser.add_argument(
    "--diss_phase",
    type=int,
    help="a delay for activation  of dissipation after the simulation begins",
    default=0,
)
parser.add_argument(
    "--cons_max",
    type=float,
    help="max value for constraints inhibition weights",
    default=-1.0,
)
parser.add_argument(
    "--activate_diss",
    type=bool,
    help="whether stimulation with Poisson sources should be activated",
    default=True,
)
parser.add_argument(
    "--stim_min", type=float, help="min value for stimulation weights", default=0.1
)
parser.add_argument(
    "--cons_min",
    type=float,
    help="min value for constraints inhibition weights",
    default=-0.5,
)
parser.add_argument(
    "--diss_delay_min",
    type=float,
    help="min value of the synaptic delay from the dissipating poisson sources to the principal neurons",
    default=1.0,
)
parser.add_argument(
    "--v_reset", type=float, help="reset membrane potential", default=-70.0
)
parser.add_argument(
    "--plot_dynamics", type=bool, help="whether plot dynamics or not", default=False
)
parser.add_argument(
    "--tau_syn_I", type=float, help="inhibitory synapse time constant", default=5.0
)
parser.add_argument(
    "--clues_inh", type=bool, help="whether allow inhibition over clues", default=True
)
parser.add_argument(
    "--diss_min", type=float, help="min value for dissipation weights", default=-1.5
)
parser.add_argument(
    "--diss_delay_max",
    type=float,
    help="max value of the synaptic delay from the dissipating poisson sources to the principal neurons",
    default=1.0,
)
parser.add_argument("--tau_m", type=float, help="membrane time constant", default=20.0)
parser.add_argument(
    "--run_time", type=int, help="duration of the simulation", default=500000
)
parser.add_argument(
    "--v_thresh", type=float, help="firing threshold voltage", default=-50.0
)
parser.add_argument(
    "--core_size",
    type=int,
    help="number of neurons per domain in a variable",
    default=42,
)
args = parser.parse_args()

name = args.name

# SpiNNaker setup.
p.setup(timestep=1.0)
# Build spiking neural network.
map = CSP(
    variables_number=193,
    domain_size=4,
    constraints=world_borders,
    core_size=args.core_size,
    run_time=args.run_time,
)
map.cell_params_lif["i_offset"] = args.i_offset
map.cell_params_lif["tau_syn_I"] = args.tau_syn_I
map.build_domains_pops()
map.internal_inhibition(
    w_range=[-abs(args.internal_min), -abs(args.internal_max)]
)  # , w_range= [-0.5,-1.0], d_range=[2.0,2.0])
map.apply_constraints(
    w_range=[-abs(args.cons_min), -abs(args.cons_max)], d_range=[2.0, 2.0]
)  # w_range=[-0.2/2.5,0.0]
map.build_stimulation_pops(
    n_populations=args.stim_pops,
    full=args.stim_full,
    shrink=args.stim_shrink,
    stim_ratio=args.stim_ratio,
    phase=args.stim_phase,
)  # , n_populations=5,shrink=0.750, stim_ratio=0.50, full=False) #, phase=15000)
map.stimulate_cores(w_range=[args.stim_max, args.stim_max], d_range=[1.0, 1.0])
map.build_dissipation_pops(
    d_populations=args.diss_pops, shrink=args.diss_shrink, phase=args.diss_phase
)  # , d_populations=2, shrink=0.250, stim_ratio=0.2, full=False, phase=args.phase)
if args.activate_diss:
    map.depress_cores(
        w_range=[-abs(args.diss_min), -abs(args.diss_max)], d_range=[1.0, 1.0]
    )
map.initialize()  # , v_range=[-65.,-55.])
# Record spikes from variable populations.
map.recording()
# Run simulation.
p.run(args.run_time)
# Save recorded spikes.
map.save(name)
map.report_network_params("report_%s" % name)
# End simulation.
p.end()
# Plot entropy.
plot_entropy(name, 200, show=True, pop_activity=args.plot_dynamics)

# python world_forsim.py
# --name world --core_size 50 --run_time 100000 --i_offset 0.0 --tau_syn_I 5.0 --internal_min -2.4 --internal_max -2.4
# --cons_min -0.5 --cons_max -1.0 --stim_pops 1
# --stim_full False --stim_shrink 1.0 --stim_ratio 1.0 --stim_phase 0 --stim_max 1.5 --stim_min 0.1
# --diss_pops 1 --diss_shrink 1.0 --diss_phase 0
# --diss_min 1.5 --diss_max 1.51

# This module has been developed by Gabriel Fonseca and Steve Furber at The University of Manchester as part of the
# paper:
#
# "Using Stochastic Spiking Neural Networks on SpiNNaker to Solve Constraint Satisfaction Problems"
# Submitted to the journal Frontiers in Neuroscience| Neuromorphic Engineering
#
""" Implement a framework to map a constraint satisfaction problem into a spiking neural network.

This module contains the CSP class, which stands for Constraint Satisfaction Problem. Its methods allow the creation of
a network of leaky integrate and fire spiking neurons whose connectivity represent the CSP problem, the connections
are either inhibitory or excitatory. The neurons are stochastically stimulated by spike sources implementing a Poisson
process causing the network dynamics to implement a stochastic search of the satisying configuration.
"""
# a separator for readability of messages on standard output
import spynnaker7.pyNN as p  # simulator

msg = '%s \n'%('='*70)

class CSP:
    """ Map a constraint satisfaction problem into a spiking neural network. """
    live = False
    run_time = 30000
    # lists for counting populations to build report.
    var_pops = []
    stim_pops = [[]]
    diss_pops = []
    n_populations = 1
    d_populations = 1
    # lists for counting synapses to build report.
    core_conns = []
    internal_conns = []
    stim_conns = []
    diss_conns = []
    stim_times = []
    diss_times = []
    constraint_conns = []
    # whether set_clues populations should receive inhibition from other sources.
    clues_inhibition = False
    # parameters for the leaky integrate and fire neurons
    cell_params_lif = {'cm': 0.25,  # nF          membrane capacitance
                       'i_offset': 0.3,  # nA          bias current
                       'tau_m': 20.0,  # ms          membrane time constant
                       'tau_refrac': 2.0,  # ms          refractory period
                       'tau_syn_E': 5.0,  # ms          excitatory synapse time constant
                       'tau_syn_I': 5.0,  # ms          inhibitory synapse time constant
                       'v_reset': -70.0,  # mV          reset membrane potential
                       'v_rest': -65.0,  # mV          rest membrane potential
                       'v_thresh': -50.0  # mV          firing threshold voltage
                       }

    def __init__(self, variables_number=0, domain_size=0, constraints=[], exc_constraints=[],
                 core_size=25, directed=False, run_time=30000):
        """ initialize the constraint satisfaction problem spiking neural network.

        Args:
            variables_number: how many variables has the CSP (integer).
            domain_size: how many values can assume each variable (integer).
            constraints: a list of tuples of conflicting variables (list of tuples).
            exc_constraints: a list of tuples of variables taking the same value (list of tuples).
            core_size: number of neurons to represent each value that a variable can assume (integer).
            directed: if True applies the constraints  from source to target as defined by the list of tuples,
                if False constraints are applied also from target to source (undirected graph).
        """
        self.variables_number = variables_number
        self.domain_size = domain_size
        self.core_size = core_size
        self.size = domain_size * core_size
        self.directed = directed
        self.constraints = constraints
        self.exc_constraints = exc_constraints
        self.clues = [[]]
        self.run_time = run_time

    def set_clues(self, clues):
        """
        Take set_clues as an array of the form [[list of variables],[list of values]].

        Here clues are fixed and predetermined values for particular variables, These influence the constraints
        implementation.

        args:
            clues: an array of the form [[list of variable ids],[list of values taken by those variables]]
        """
        self.clues = clues


    def build_domains_pops(self):
        """
        generate an array of pyNN population objects, one for each CSP variable.

        The population size will be self.size = domain_size*core_size.
        var_pops[i] is the population for variable i including all domain sub-populations, each of zise core_size.
        """
        print(msg, 'creating an array of %d neural populations'%(self.variables_number))
        var_pops = []
        for variable in range(self.variables_number):
            var_pops.append(p.Population(self.size,
                                         p.IF_curr_exp, self.cell_params_lif,
                                         label="var%d" % (variable + 1)))
        self.var_pops = var_pops

    def poisson_params(self, n_populations, full=False, stim_ratio=1.0, shrink=1.0, phase=0.0):
        """ 
        Define time intervals for activation of the pyNN Poisson noise sources.

        This method defines the temporal dependence of the stimulating noise, it is an internal method called by the 
        build_stimulation_pops method. Here we use the word noise to refer to spike sources implementing a random 
        Poisson process. In pyNN these objects connect with neurons using synapses as if they were neurons too.
        Currently each SpikeSourcePoisson object accepts only a start time and a duration time, thus to change the noise
        level through time one should create n different populations and activate them at different times. 
        Here we uniformly distribute the start times of the n_populations from phase to run_time. Each population will be active 
        for a period lapso = shrink * self.run_time / n_populations if full=False otherwise will stay active during all 
        run_time. To avoid synchronization of all the noise sources and improve the stochasticity of the search a
        time interval delta is defined to randomly spread the activation 
        and deactivation times of the SpikeSourcePoisson objects, see diagram.

                   |comienza-|-------Full Noise--------|--termina|
                   |--delta--|                         |--delta--|
                   0%-------100%                      100%------0%
                   |------Noise=(lapso - stim_ratio)---|--stim_ratio--------|            
                   |------lapso=runtime*shrink/n_popultions--------------...|
        |--phase---|------noise interval = runtime*shrink----------------------------------------...|
        |-----------------------------------------------run_time----------------------------------------------------...|
        
        Other stochastic search strategies to solve the CSP may be implemented modifying this method and the 
        build_stimulation_pops method below.        


        args:
            n_populations: number of noise populations to stimulate each CSP variable. 
            shrink: shrink the run_time to uniformelly distribute noise pulses in a smaller interval (from 0. to 1.)
            full: controls if the stimulations deactivation should all happen after run_time or at the lapso width.
            stim_ratio: defines the portion of the stimulation window in which the stimulation will be active. A value 
                of 0.5 will mean that stimulation happens only during the first half of the interval.

        returns: 
            list of starting times
            lists of random distributions for start and duration of noise stimulation populations.
        """
        lapso = shrink * self.run_time / n_populations
        delta = lapso / self.run_time
        comienza = [RandomDistribution("uniform", [lapso * i+phase, lapso * i + delta+phase]) for i in range(n_populations)]
        if full:
            termina = [RandomDistribution("uniform", [self.run_time - delta, self.run_time]) for i
                       in range(n_populations)]
        else:
            termina = [RandomDistribution("uniform", [lapso * stim_ratio, lapso * stim_ratio + delta]) for i
                       in range(n_populations)]
        stim_times = [lapso * i+phase for i in range(n_populations)]
        return stim_times, comienza, termina

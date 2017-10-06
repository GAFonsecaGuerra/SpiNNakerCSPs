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
from pyNN.random import RandomDistribution

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
            full: controls if the noise deactivations should all happen after run_time or at the lapso width.
            stim_ratio: defines the portion of the stimulation window in which the noise will be active. A value
                of 0.5 will mean that stimulation happens only during the first half of the interval.
            shrink: shrinks the time interval throughout wich the noise populations will be distributed.
                It defines fraction of the run time, so it should be between 0.0 and 1.0.
            phase: a waiting time before the first noise population activates.
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

    def build_stimulation_pops(self, n_populations=1, shrink=1.0, stim_ratio=1.0, rate=(20.0, 20.0), full=True,
                               phase=0.0, clue_size=None):
        """ Generate noise sources for each neuron and creates additional stimulation sources for clues.

        The noise sources are pyNN population objects of the SpikeSourcePoisson type, which generate spikes at times
        described by a Poisson random process. In pyNN these objects connect with neurons using synapses as if they
        were neurons too, these will be excitatorily  connected to the variable populations as stimulating noise.

        Currently each SpikeSourcePoisson object accepts only a start time and a duration time, thus to change the noise
        level through time one should create n different populations and activate them at different times. This method
        passes the arguments to the poisson_params method and reads activation times and duration from its returns.
        Such times define the stochastic search.

        args:
            n_populations: number of noise populations to stimulate each CSP variable.
            shrink: shrinks the time interval throughout wich the noise populations will be distributed.
                It defines fraction of the run time, so it should be between 0.0 and 1.0.
            stim_ratio: defines the portion of the stimulation window in which the noise will be active. A value
                of 0.5 will mean that stimulation happens only during the first half of the interval.
            rate: a tuple of floating-point numbers defining the rate of the Poisson process for the noise
                  populations, the first value is used for all CSP variable populations and the second value for
                  the clues.
            full: controls if the noise deactivations should all happen after run_time or at the lapso width.
            phase: a waiting time before the first noise population activates.
            clue_size: optional, number of neurons to use to stimulate clues, default value is core_size.
        """
        print(msg, 'creating %d populations of SpikeSourcePoisson noise sources for each variable'%(n_populations))
        stim_times, comienza, termina = self.poisson_params(n_populations, full=full, stim_ratio=stim_ratio,
                                                            shrink=shrink, phase=phase)
        if clue_size == None:
            clue_size = self.core_size
        stim_pops = [[] for k in range(n_populations)]
        clues_stim = []
        for stimulus in range(n_populations):
            for variable in range(self.variables_number):
                stim_pops[stimulus].append(p.Population(self.size, p.SpikeSourcePoisson,
                                                 {"rate": rate[0], "start": comienza[stimulus], "duration":
                                                     termina[stimulus]}, label="stim%d_var%d" % (stimulus + 1, variable
                                                                                                 + 1)))
                if variable in self.clues[0]:
                    clues_stim.append(p.Population(clue_size, p.SpikeSourcePoisson,
                                                   {"rate": rate[1], "start": 0, "duration": self.run_time},
                                                   label='clues_stim%d' % variable))
        self.stim_pops = stim_pops
        self.clues_stim = clues_stim
        self.n_populations = n_populations
        self.stims = stim_times
        self.clue_size = clue_size

    def build_dissipation_pops(self, d_populations=1, shrink=1.0, stim_ratio=1.0, rate=20.0, full=True, phase=0.0):
        """ Generate noise sinks for each neuron: pyNN population objects of the type SpikeSourcePoisson.

        the Poisson neural populations will be inhibitorilly connected to the variable populations, creating a
        dissipative effect. This method passes the arguments to the poisson_params method and reads activation
        times from the returns. If the clues_inhibition argument is False the clues will not receive inhibition.

        args:
            d_populations: number of dissipative noise populations to depress each CSP variable.
            shrink: shrinks the time interval throughout wich the noise populations will be distributed.
                It defines fraction of the run time, so it should be between 0.0 and 1.0.
            stim_ratio: defines the portion of the depression window in which the noise will be active. A value
                of 0.5 will mean that depression happens only during the first half of the interval.
            rate: a floating-point number defining the rate of the Poisson process for the noise
                  populations.
            full: controls if the noise deactivations should all happen after run_time or at the lapso width.
            phase: a waiting time before the first dissipation population activates.
        """
        print(msg, 'creating %d populations of dissipative noise sources for each variable' % (d_populations))
        diss_times, comienza, termina = self.poisson_params(d_populations, full=full, stim_ratio=stim_ratio,
                                                            shrink=shrink, phase=phase)
        diss_pops = [[] for k in range(d_populations)]
        for k in range(d_populations):
            for variable in range(self.variables_number):
                diss_pops[k].append(p.Population(self.size, p.SpikeSourcePoisson,
                                                 {"rate": rate, "start": comienza[k], "duration": termina[k]},
                                                 label="diss%d_var%d" % (k + 1, variable + 1)))
        #TODO: if self.clues_inibition = False do not create the populations for the clues.
        self.diss_pops = diss_pops
        self.d_populations = d_populations
        self.disss = diss_times

   def connect_cores(self, w_range=[0.6, 1.2], d_range=[1.0, 1.2]):
        """
        Create internal excitatory connections between the neurons of each domain subpopulation of each variable.

        In the network representing the CSP, each neural population representing a variable contains a subpopulation
        for each possible value on its domain. This method connects all-to-all the neurons of each domain population
        of each variable using escitatory synapses.

        args:
            w_range: range for the random distribution of synaptic weights in the form [w_min, w_max].
            d_range: range for the random distribution of synaptic delays in the form [d_min, d_max].
        """
        print(msg, 'internally connnecting the neurons of each domain of each variable')
        delays = RandomDistribution('uniform', d_range)
        weights = RandomDistribution('uniform', w_range)
        connections = [(m, n, weights.next() if m // self.core_size == n // self.core_size and m != n else 0.0,
                        delays.next()) for n in range(self.domain_size * self.core_size) for m in
                       range(self.domain_size * self.core_size)]
        for variable in range(self.variables_number):
            synapses = p.Projection(self.var_pops[variable], self.var_pops[variable], p.FromListConnector(connections,
                                                                                                          safe=True),
                                    target="excitatory")
            self.core_conns.append(synapses)

    def internal_inhibition(self, w_range=[-0.2, 0.0], d_range=[2.0, 2.0]):
        """
        Connect the domains populations of the same variable using inhibitory synapses.

        the connectiviy establishes a lateral inhibition circuit over the domains of each variable, so that most of the
        time only the neurons from a single domain population are active.

        args:
            w_range: range for the random distribution of synaptic weights in the form [w_min, w_max].
            d_range: range for the random distribution of synaptic delays in the form [d_min, d_max].
        """
        print(msg, 'Creating lateral inhibition between domains of each variable')
        delays = RandomDistribution('uniform', d_range)
        weights = RandomDistribution('uniform',w_range)
        connections = [(m, n, 0.0 if m // self.core_size == n // self.core_size else  weights.next(), delays.next()) for
                       n in range(self.size) for m in range(self.size)]
        for variable in range(self.variables_number):
            if self.clues_inhibition:
                synapses = p.Projection(self.var_pops[variable], self.var_pops[variable], p.FromListConnector(connections, safe=True),
                                        target="inhibitory")
                self.internal_conns.append(synapses)
            elif variable not in self.clues:
                synapses = p.Projection(self.var_pops[variable], self.var_pops[variable], p.FromListConnector(connections, safe=True),
                                        target="inhibitory")
                self.internal_conns.append(synapses)


    def stimulate_cores(self, w_range=[1.4, 1.4], d_range=[1.0, 1.0], w_clues=[1.4, 1.6]):  # w_clues=[0.0, 0.2]
        """ connect stimulating noise sources to variables populations.

        args:
            w_range: range for the random distribution of synaptic weights in the form [w_min, w_max].
            d_range: range for the random distribution of synaptic delays in the form [d_min, d_max].
            w_clues: clues specific range for the random distribution of synaptic weights in the form [w_min, w_max].
        """
        p.set_number_of_neurons_per_core(p.IF_curr_exp, 150)
        print(msg, 'connecting Poisson noise sources to neural populations for stimulation')
        delays = RandomDistribution('uniform', d_range)
        weights = RandomDistribution('uniform', w_range)
        weight_clues = RandomDistribution("uniform", w_clues)
        for stimulus in range(self.n_populations):
            for variable in range(self.variables_number):
                counter = 0
                if variable in self.clues[0]:
                    shift = self.clues[1][self.clues[0].index(variable)] * self.core_size
                    connections = [(m, n + shift, weight_clues.next(), delays.next()) for m in range(self.core_size) for
                                   n in range(self.clue_size)]
                    synapses = p.Projection(self.clues_stim[counter], self.var_pops[variable],
                                            p.FromListConnector(connections, safe=True), target='excitatory')
                    counter += 1
                    self.stim_conns.append(synapses)
                else:
                    synapses = p.Projection(self.stim_pops[stimulus][variable], self.var_pops[variable],
                                            p.OneToOneConnector(weights=weights, delays=delays), target='excitatory')
                    self.stim_conns.append(synapses)
        self.stim_times += self.stims

    def depress_cores(self, w_range=[-2.0, -1.5], d_range=[2.0, 2.0]):
        """ connect depressing noise sources to variables populations.

        args:
            w_range: range for the random distribution of synaptic weights in the form [w_min, w_max].
            d_range: range for the random distribution of synaptic delays in the form [d_min, d_max].
        """
        print(msg, 'connecting Poisson noise sources to neural populations for dissipation')
        delays = RandomDistribution('uniform', d_range)
        weights = RandomDistribution('uniform', w_range)
        for depressor in range(self.d_populations):
            for variable in range(self.variables_number):
                if variable not in self.clues[0]:
                    synapses = p.Projection(self.diss_pops[depressor][variable], self.var_pops[variable],
                                            p.OneToOneConnector(weights=weights, delays=delays), target='inhibitory')
                    self.diss_conns.append(synapses)
        self.diss_times += self.disss





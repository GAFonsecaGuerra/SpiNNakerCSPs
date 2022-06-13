# This module has been developed by Gabriel Fonseca and Steve Furber at The University of Manchester as part of the
# paper:
#
# "Using Stochastic Spiking Neural Networks on SpiNNaker to Solve Constraint Satisfaction Problems"
# Submitted to the journal Frontiers in Neuroscience| Neuromorphic Engineering
# -----------------------------------------------------------------------------------------------------------------------
"""Implement a set of functions to analyse the spikes generated by the simulation and generate plots.

This module contains a set of functions, to obtain and plot the temporal dependence of the shanon entropy, average
firing rate, visited configurations and domain subpopulations activity from the spikes file generated in the simulation
of the spiking neural network representation of the constraint satisfaction problem.
"""
import numpy as np
import simplejson as sj
import os
from random import randint
from scipy.interpolate import UnivariateSpline

# Check whether matplotlib is available for importing or continue without plotting.
missing_matplotlib = False
try:
    import matplotlib.pyplot as plt

    plt.ioff()
except:
    print("matplotlib is not present, continuing without plots")
    missing_matplotlib = True

msg = "%s \n" % (
    "=" * 70
)  # A separator for readability of messages on standard output.


def plot_entropy(
    spikes_file=None,
    resolution=200,
    save_to=False,
    show=True,
    draw_pulses=True,
    cumulative=True,
    cumulative_decay=0.97,
    splines=True,
    pop_activity=True,
    cons_type="inhibitory",
    preprint=True,
    font=18,
    rate_units="k spikes",
    plots=True,
    xrange=None,
    var_to_plot=None,
    TSP=False,
    lw=3,
):
    """Generate plot of Shannon entropy, firing rate, visited states and populations activity all vs. time.

    the entropy plot is colorized according to satisfiability: Red=False, Blue = True, Green = False and configuration
    already visited. The plot of visited states is black for new configurations and yellow for configurations already
    visited. The activity plots represent the activity of the competing domains in a given variable(s) the plot for each
    domain will be plotted in a different color.

    This function uses auxiliary functions defined below to:
            -Get spikes from recorded files.
            -Organize spikes as a 3Darray of the form (dom_id)X(var_id)X(spike_times) i.e. a list of lists.
            -Pool spikes in time probability bins.
            -Normalize probability bins to the total num of spikes (for all digits) in each time bin.
            -Find most likely domain for each variable at each time bin.
            -Count visited configurations vs. time bins.
            -Check satisfiability of constraints for each time bin.
            -Get the network average firing rate at each time bin
            -Compute entropy S=Sum_{digits}[-p{digit}ln(p{digit})] for each time bin generating an array of the form
             (entropy)vs(DeltaT).
            -Plot entropy vs. time coloured with satisfiability.
            -Plot firing rate vs. time.
            -Plot Omega vs. time coloured with repeating states.
            -Generate plots of the activity of the competing domains in a given set of variable.

    args:
        spikes_file: where the spikes are saved.
        resolution: the size of the time bins in milliseconds.
        save_to: name of file to save the entropy plot and .dat data.
        show: whether open the figures generated
        draw_pulses: whether draw Blue (Red) vertical lines at stimulating (depressing) noise activation times.
        cumulative: whether accumulate probabilities over the entire run.
        cumulative_decay: decay rate for old cumulative contributions.
        splines: wheter include a spline fit with the activity plots.
        pop_activity: whether generate plots of the activity of the competing domains in a given variable.
        cons_type: inhibitory or excitatory for the synapse type used to implement constraints.
        preprint: whether include titles.
        font: font size for labels.
        rate_units: 'spikes' or 'k spikes' affect the y axis of the firing rate plot.
        plots: whether generate plots or not.
        xrange: range of the horizontal axis.
        var_to_plot: plot the domains activity of these variables.It could be a list of the form [v_min, v_max],
            None, the number of the variable as an integer or 'All' to generate plots for all variables.
        TSP: whether save the TSP_winners_matrix to a file for travelling salesman problems.
        lw:line width to be used in the plots.

    returns:
        plots and lists of Shannon entropy, firing rate and visited states vs time.
    """
    spikes, params = load_data(spikes_file)
    (
        run_time,
        variables_number,
        size,
        domain_size,
        core_size,
        constraints,
        stim_times,
        diss_times,
    ) = params
    time_bins = int(run_time / resolution)
    probabilities, spikecount, pT = compute_probabilities(
        spikes,
        variables_number,
        domain_size,
        core_size,
        time_bins,
        resolution,
        cumulative,
        cumulative_decay,
    )
    H, max_H, p_max = compute_entropy(
        probabilities, variables_number, domain_size, time_bins
    )
    satisfiability = constraint_checker(p_max, constraints, cons_type)
    change = state_change_checker(p_max)
    is_state_new, visited_states = new_state(p_max)
    if save_to == False:
        save_to = spikes_file
    if not missing_matplotlib:
        if plots:
            entropy_plot(
                H,
                time_bins,
                run_time if xrange == None else xrange,
                resolution,
                satisfiability,
                is_state_new,
                change,
                pT,
                draw_pulses,
                stim_times,
                diss_times,
                save_to,
                max_H,
                show,
                visited_states,
                preprint,
                font,
                rate_units,
            )
    # save entropy vs. time plot and list to local files
    if TSP:
        with open("TSP_winners_matrix", "w") as file:
            sj.dump(p_max, file)
    with open("results/entropy_%s.txt" % save_to, "w+") as file:
        np.savetxt(file, [H, satisfiability])
    # ------------------------- pop activity
    if not missing_matplotlib:
        if pop_activity:
            plot_pop_activity(
                spikecount,
                time_bins,
                variables_number,
                domain_size,
                stim_times,
                diss_times,
                draw_pulses,
                save_to,
                splines,
                font=font,
                resolution=resolution,
                run_time=run_time if xrange == None else xrange,
                var_to_plot=var_to_plot,
                show=show,
                lw=lw,
            )
    sol, sol_time = get_solution(p_max, satisfiability)
    if sol_time is not None:
        print(
            msg,
            """The simulation found a solution for the first time
        at time %d,
        at state %d,
        the total number of states visited is:%d
        """
            % (sol_time * resolution, visited_states[sol_time + 1], visited_states[-1]),
        )
    return sol


def load_data(prefix=None):
    """Get spikes from file.

    args:
        prefix: the prefix of the name of the file containing the spikes to be processed the file name will be of
            the form prefix_spikes_binary.
    """
    # with open("results/%s_spikes_binary.npy" % (prefix)) as database:
    npyfile = np.load("results/%s_spikes_binary.npz" % (prefix), allow_pickle=True)
    params = npyfile["params"]
    (
        run_time,
        variables_number,
        size,
        domain_size,
        core_size,
        constraints,
        stim_times,
        diss_times,
    ) = params
    spikes = [[] for variable in range(variables_number)]
    for variable in range(variables_number):
        spikes[variable] = npyfile["arr_%s" % (variable)]
    return spikes, params


def compute_probabilities(
    spikes,
    variables_number,
    domain_size,
    core_size,
    time_bins,
    resolution,
    cumulative,
    cumulative_decay,
):
    """Process spikes into firing probabilities using time bins.

    args:
        spikes: the array of firing times from each neuron, it is extracted from a file with load_data(file).
        variables_number: number of variables of the CSP.
        domain_size: number of possible values in the domain of each variable.
        core_size: number of neurons used to represent each domain of each variable.
        time_bins: the number of time bins.
        resolution: the size of each time bin.
        cumulative: whether probabilities are made cummulative or not.
        cumulative_decay: decay rate for old cumulative contributions to the probability.

    returns:
        probabilities: probability matrix, probability per domain per time bin per variable.
        spikecount: spikes per time bin per domain per variable.
        pT: array of the sum of the probabilities of all domains, for each variable and for each time.
    """
    probabilities = [
        [[0.0 for dom in range(domain_size)] for t in range(time_bins)]
        for var in range(variables_number)
    ]
    spikecount = [
        [[0.0 for t in range(time_bins)] for dom in range(domain_size)]
        for var in range(variables_number)
    ]
    pT = [[0.0 for i in range(time_bins)] for var in range(variables_number)]
    max_spikes = 0
    # Count spikes per domain per time bin per variable., write to a probability matrix.
    for var in range(variables_number):
        for spike in spikes[var]:
            probabilities[var][int(spike[1] / resolution)][
                int(spike[0] / core_size)
            ] += 1
            spikecount[var][int(spike[0] / core_size)][int(spike[1] / resolution)] += 1
        for time in range(time_bins):
            pT[var][time] = sum(probabilities[var][time])
            if pT[var][time] > max_spikes:
                max_spikes = pT[var][time]

        if cumulative:
            for dom in range(domain_size):
                for bin in range(time_bins - 1):
                    probabilities[var][bin + 1][dom] += (
                        probabilities[var][bin][dom] * cumulative_decay
                    )
        for bin in range(time_bins):
            p_tot = sum(
                probabilities[var][bin]
            )  # Total number of spikes per time bin per variable.
            for dom in range(domain_size):
                if p_tot != 0.0:
                    probabilities[var][bin][dom] = probabilities[var][bin][dom] / p_tot
                else:
                    probabilities[var][bin][dom] = None
    return probabilities, spikecount, pT


def compute_entropy(probabilities, variables_number, domain_size, time_bins):
    """Compute total entropy per time_bin and find the winner digit for each variable per time bin.

    args:
        probabilities: the probability matrix generated by compute_probabilities, probability per domain per time bin
            per variable.
        variables_number: number of variables of the CSP.
        domain_size: number of possible values in the domain of each variable.
        time_bins: the number of time bins.

    returns:
        H: entropy list a value per each time bin.
        max_H: theoretical maximum entropy.
        p_max: probability of the winner domain per time bin for each variable (array).
    """
    H = [0.0 for i in range(time_bins)]
    bits = 1 / np.log(2.0)  # for conversion to log2
    max_H = (
        variables_number * np.log(domain_size) * bits
    )  # maximum possible entropy - all nos equally likely
    p_max = [[[0, 0.0] for bin in range(time_bins)] for var in range(variables_number)]
    for t in range(time_bins):
        for var in range(variables_number):
            for dom in range(domain_size):
                # normalize probabilities with regard to the maximum entropy on each time bin
                if probabilities[var][t][dom] > p_max[var][t][1]:
                    p_max[var][t] = [dom, probabilities[var][t][dom]]
                if probabilities[var][t][dom] > 0.0:
                    H[t] += (
                        -probabilities[var][t][dom]
                        * np.log(probabilities[var][t][dom])
                        * bits
                    )
    return H, max_H, p_max


def constraint_checker(winners_matrix, constraints, cons_type):
    """Check satisfiability of all constraints at each time_bin.

    args:
        winners_matrix: a matrix of the values taken by each variable at each time_bin. Each value in the matrix is of
            the form [value, probability] where the probability is measured with regard to the other possible values.
            Satisfiability will be checked for each time_bin.
        constraints: list of constraints defining the CSP.

    returns:
        satisfiability: array of truth values of satisfiability for each time bin.
    """
    satisfiability = []
    time_bins = np.shape(winners_matrix)[1]
    for time in range(time_bins):
        # violations = 0
        truth_value = None
        for constraint in constraints:
            source = constraint["source"]
            target = constraint["target"]
            if cons_type == "inhibitory":
                if (
                    winners_matrix[source][time][0] != winners_matrix[target][time][0]
                ):  # winners_matrix[variable][time][[val, prob]]
                    truth_value = True
                else:
                    truth_value = False
                    # Break the code at first violated constraint,
                    # for now we disregard the number of constraint violations.
                    break  # violations += 1
            if cons_type == "excitatory":
                if (
                    winners_matrix[source][time][0] == winners_matrix[target][time][0]
                ):  # winners_matrix[variable][time][[val, prob]]
                    truth_value = True
                else:
                    truth_value = False
                    # Break the code at first violated constraint,
                    # for now we disregard the number of constraint violations.
                    break  # violations += 1
        satisfiability.append(
            truth_value
        )  # satisfiability.append([truth_value, violations])
    return satisfiability


def state_change_checker(winners_matrix):
    """Check if the state of the system changed between two consecutive times t and t'.

    State here is defined with respect to the firing of spikes and not to the membrane potential value.

    args:
        winners_matrix: a matrix of the values taken by each variable at each time_bin. Each value in the matrix is of
            the form [value, probability] where the probability is measured with regard to the other possible values.

    returns:
        state_change: array of truth value for change of state at each time bin.
    """
    state_change = [True]
    time_bins = np.shape(winners_matrix)[1]
    number_variables = np.shape(winners_matrix)[0]
    for time in range(time_bins - 1):
        truth_value = None
        for variable in range(number_variables):
            if (
                winners_matrix[variable][time][0]
                == winners_matrix[variable][time + 1][0]
            ):
                truth_value = False
            else:
                truth_value = True
                break
        state_change.append(truth_value)
    return state_change


def get_solution(winners_matrix, satisfiability):
    """If found, get the solution and the time at which it was found.

    args:
        winners_matrix: a matrix of the values taken by each variable at each time_bin. Each value in the matrix is of
            the form [value, probability] where the probability is measured with regard to the other possible values.
        satisfiability: array of truth values of satisfiability for each time bin.

    returns:
        solution: assignation of values for the CSP variables such that all constraints are satisfiesd.
        solution_time: time at which the network found the solution for the first time.
    """
    solution = None
    solution_time = None
    for time, truth_value in enumerate(satisfiability):
        if truth_value == True:
            solution = [var[time][0] for var in winners_matrix]
            solution_time = time
            break
    if solution_time == None:
        last_bin = len(satisfiability) - 1
        print(
            msg,
            "the system did not find a solution, the last configuration of the network was:",
        )
        last = [var[last_bin][0] for var in winners_matrix]
        print(last)
    else:
        print(
            msg,
            "The simulation found a solution for the first time at bin %d"
            % solution_time,
        )
    return solution, solution_time


def new_state(winners_matrix):
    """Check if the current state of the neural network has been visited at any previous time.

    State here is defined with respect to the firing of spikes and not to the value of the membrane potential.

    args:
        winners_matrix: a matrix of the values taken by each variable at each time_bin. Each value in the matrix is of
            the form [value, probability] where the probability is measured with regard to the other possible values.

    returns:
        newness: array of truth values one per time bin, True if the state is new (never visited before) and False
            otherwise.
        visited_states: array with the number of configurations that have been visited at each time.
    """
    newness = []
    time_bins = np.shape(winners_matrix)[1]
    variables_number = np.shape(winners_matrix)[0]
    visited_configurations = (
        []
    )  # Will save the configuration of the network at each time.
    visited_states = [
        0
    ]  # Will save the number of configurations that have been visited at each time.
    for time in range(time_bins):
        is_state_new = None
        configuration = [
            winners_matrix[variable][time][0] for variable in range(variables_number)
        ]
        if configuration in visited_configurations:
            is_state_new = False
            visited_states.append(visited_states[-1])
        else:
            is_state_new = True
            visited_states.append(visited_states[-1] + 1)
        visited_configurations.append(configuration)
        newness.append(is_state_new)
    return newness, visited_states


def entropy_plot(
    H,
    time_bins,
    run_time,
    resolution,
    satisfiability,
    newness,
    state_change,
    pT,
    draw_pulses,
    stim_times,
    diss_times,
    save_to,
    max_H,
    show,
    visited_states,
    preprint,
    font,
    rate_units,
):
    """Plot entropy vs. time.

    args:
        H: entropy list a value per each time bin.
        time_bins: the number of time bins.
        run_time: duration of the simulation.
        resolution: the size of the time bins in milliseconds.
        satisfiability: array of truth values of satisfiability for each time bin.
        newness: array of truth values one per time bin, True if the state is new (never visited before) and False
            otherwise.
        state_change: array of truth value for change of state at each time bin.
        pT: array of the sum of the probabilities of all domains, for each variable and for each time.
        draw_pulses: whether draw Blue (Red) vertical lines at stimulating (depressing) noise activation times.
        stim_times: activation times of the stimulating noise pulses.
        diss_times: activation times of the depressing noise pulses.
        save_to: prefix of the file name for the entropy plot, the full name will be prefix_entropy.png.
        max_H: theoretical maximum entropy.
        show: whether open the figures generated
        visited_states: array with the number of configurations that have been visited at each time.
        preprint: whether include titles.
        font: font size for labels.
        rate_units: 'spikes' or 'k spikes' affect the y axis of the firing rate plot.
    """
    plt.figure(1, figsize=[8, 10])
    plt.tick_params(axis="both", which="major", labelsize=30)
    plt.subplot(311)  # numrows, numcols, fignum.
    for time in range(time_bins):
        state_change_color = "r." if state_change[time] else "g."
        plt.plot(
            time * resolution,
            H[time],
            "b." if satisfiability[time] else state_change_color,
        )
        plt.plot(time * resolution, -1.0, "k." if newness[time] else "y.")
    if draw_pulses:
        for i in stim_times:
            plt.axvline(x=i, color="b", linestyle="dashed")
        for i in diss_times:
            plt.axvline(x=i, color="r", linestyle="dashed")
    if preprint:
        plt.title(save_to)
    plt.ylabel(r"$H$(bits)", fontsize=font)
    # plt.xlabel('time (ms)',fontsize=font)
    plt.axis([-2, run_time, -10, max(H) + 20])
    plt.tick_params(axis="both", which="major", labelsize=15)

    plt.subplot(312)
    conversor = {"k spikes": 1000, "spikes": 1}
    nu = np.divide(np.array(pT).sum(0), conversor[rate_units])
    times = [i * resolution for i in range(time_bins)]
    plt.plot(times, nu, "k-")
    plt.ylabel(r"$\nu$ (%s/s)" % (rate_units), fontsize=font)
    # plt.axis([-2, run_time, -10, max(H)+20])
    plt.tick_params(axis="both", which="major", labelsize=15)
    # plt.xlabel('time (ms)', fontsize=font)

    plt.subplot(313)
    for time in range(time_bins):
        plt.plot(
            time * resolution, visited_states[time], "k." if newness[time] else "y."
        )
    plt.ylabel(r"$\Omega$", fontsize=font)
    plt.xlabel("time (ms)", fontsize=font)
    plt.axis([-2, run_time, -max(visited_states) * 0.2, max(visited_states) * 1.3])
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.savefig("results/%s_entropy.png" % save_to)
    if show:
        plt.show()
    plt.close()
    return


def plot_pop_activity(
    spikecount,
    time_bins,
    variables_number,
    domain_size,
    stim_times,
    diss_times,
    draw_pulses,
    save_to,
    splines=True,
    font=20,
    noticks=False,
    resolution=None,
    run_time=None,
    var_to_plot=None,
    show=True,
    lw=1.0,
):
    """Generate plots of the activity of the competing domains in a given set of variables.

    The plot for each domain will be plotted in a different color.

    args:
        spikecount: spikes per time bin per domain per variable.
        time_bins: the number of time bins.
        variables_number: number of variables of the CSP.
        domain_size: number of possible values in the domain of each variable.
        stim_times: activation times of the stimulating noise pulses.
        diss_times: activation times of the depressing noise pulses.
        draw_pulses: whether draw Blue (Red) vertical lines at stimulating (depressing) noise activation times.
        splines: wheter include a spline fit with the activity plots.
        save_to: name of file to save the activity plots.
        font: font size for labels.
        noticks: whether axes ticks should be used or not.
        resolution: the size of the time bins in milliseconds.
        run_time: duration of the simulation.
        var_to_plot: plot the domains activity of these variables.It could be a list of the form [v_min, v_max],
            None, the number of the variable as an integer or 'All' to generate plots for all variables.
        show: whether open the figures generated.
        lw:line width to be used in the plots.
    """
    if not os.path.exists("results/Dynamics"):
        os.makedirs("results/Dynamics")
    pop_activity = [
        [[0.0 for time in range(time_bins)] for domain in range(domain_size)]
        for variable in range(variables_number)
    ]
    for variable in range(variables_number):
        for domain in range(domain_size):
            for time in range(time_bins):
                if spikecount[variable][domain][time] != 0.0:
                    pop_activity[variable][domain][time] = spikecount[variable][domain][
                        time
                    ] / max(spikecount[variable][domain])
                else:
                    pop_activity[variable][domain][time] = 0.0
    if isinstance(var_to_plot, int):
        var_to_plot = [var_to_plot]
    elif var_to_plot == None:
        var_to_plot = [randint(0, variables_number - 1)]
    elif var_to_plot == "All":
        var_to_plot = range(variables_number)
    counter = 0
    for variable in var_to_plot:
        plt.figure()
        if noticks:
            plt.xticks([])
            plt.yticks([])
        colors = ["r", "b", "g", "c", "m", "y", "k", "0.75", "burlywood"]
        times = np.multiply(
            list(np.arange(0, time_bins)), resolution
        )  # verify if it is necesary to be a list
        for domain in range(domain_size):
            plt.plot(
                times,
                pop_activity[variable][domain],
                ".",
                color=colors[domain],
                markersize=lw * 2,
            )
            if splines:
                spl = UnivariateSpline(times, pop_activity[variable][domain])
                spl.set_smoothing_factor(0.10)  # 3
                xs = np.linspace(0, run_time, 10000)
                plt.plot(xs, spl(xs), color=colors[domain], linewidth=lw)
            plt.axis([-2, run_time, 0, 1])
        if draw_pulses:
            for time in stim_times:
                plt.axvline(x=time, color="b", linestyle="dashed")
            for time in diss_times:
                plt.axvline(x=time, color="r", linestyle="dashed")
        plt.ylabel("$\hat{A}$", fontsize=font)
        plt.xlabel("time (ms)", fontsize=font)
        plt.tick_params(axis="both", which="major", labelsize=15)
        plt.savefig(
            "results/Dynamics/%s_dynamics_variable_%d.png"
            % (save_to, var_to_plot[counter])
        )  # save plot
        if show:
            plt.show()
        plt.close()
        counter += 1

"""Plot performance of classifiers for simulation data.

This program measures and reports the AUC of the FDensemble, Woc, and best 
individual classifiers for NREPS replicate simulation data and differing 
simulation parameter values.  The different simulation parameters that are 
varied are the number of base classifiers M and the class conditioned 
correlation between base classifier predictions.  Each classifier is trained, 
if training is necessary, and its AUC measured on the same data.

The plots that report the simulation results are printed as pdf's in the directory 
'sim_performance'.  This directory is assumed to be located in the same directory
as this file.  If the directory does not exists in the expected location, it is 
automatically made by this program.
"""

from os import mkdir, path

import numpy as np
import matplotlib.pyplot as plt

import utilities as utils

from fd import stats, sample
from fd import classifiers as cls

# simulation parameters
M = np.array([2, 4, 7, 10, 15, 20, 25])
N = 500
PREVALENCE = 0.4
CORRELATIONS = np.array([0, 0.2, 0.4, 0.6])
NREPS = 10
AUC_LIMITS = (0.55, 0.75)

# plot parameters
COLORS = {"FiDEL": "blue",
          "WOC": "orange",
          "Best Ind": "green"}
DIRECTORY="sim_performance"


def plot_errorbars(items, *, xlabel, ylabel, savename):
    """Plot line plot with error bars and print to file.

    Args:
        items: (dictionary) of utils.DataForPlots class instances, each
            dictionary item represents a distinct line and set of 
            points to be plotted.
        xlabel: (str) label for x-axis
        ylabel: (str) label for y-axis
        savename: (str) name of file plot should be printed to
    """
    plt.figure(figsize=(5,3))
    for item in items.values():
        plt.errorbar(item.x, item.y, 
                     xerr=item.xerr, yerr=item.yerr,
                     fmt="o", capsize=2.5, ms=10, mfc="none",
                     mec=item.color, label=item.label)

    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 0.8, 0.2, 0.2))

    for item in items.values():
        plt.plot(item.x, item.y, "-", color=item.color)

    ax = plt.gca()
    ax.set_position([0.175, 0.2, 0.55, 0.75])

    plt.ylim(0.65, 1.01)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.savefig(savename, fmt=savename.split(".")[-1])


def main():

    # if directory for figures does not exists, make it
    if not path.exists(path.join(path.dirname(__file__), DIRECTORY)):
        mkdir(path.join(path.dirname(__file__), DIRECTORY))

    # initialize dictionaries for per classifier items

    # classifier class instances for training and evaluating performance
    classifiers = {"FiDEL": cls.FDensemble(),
                   "WOC": cls.Woc(),
                   "Best Ind": cls.BestInd()}

    # keep per classifier performance statistics
    clstats = {"FiDEL": utils.StatsTable(CORRELATIONS.size, 
                                   M.size),
               "WOC": utils.StatsTable(CORRELATIONS.size, 
                                 M.size),
               "Best Ind": utils.StatsTable(CORRELATIONS.size, 
                                      M.size)}

    corr_stats = utils.StatsTable(CORRELATIONS.size,
                                M.size)

    # keep per classifier data for plotting
    clplots = {"FiDEL": utils.DataForPlots("FiDEL", COLORS["FiDEL"]),
               "WOC": utils.DataForPlots("WOC", COLORS["WOC"]),
               "Best Ind": utils.DataForPlots("Best Ind.", COLORS["Best Ind"])}


    for i, corr in enumerate(CORRELATIONS):

        for j, m in enumerate(M):

            base_classifiers_auc = utils.get_auc(m, AUC_LIMITS)

            # inititialize data structure which keeps per classifier
            # replicate performance data and computes statistics
            clreps = {"FiDEL": utils.ReplicateData(NREPS),
                      "WOC": utils.ReplicateData(NREPS),
                      "Best Ind": utils.ReplicateData(NREPS)}

            # initialize data structure for storing correlation values
            corr_reps = utils.ReplicateData(NREPS)

            # Generate NREPS synthetic data to train and evaluate classifier
            # performance
            for nrep in range(NREPS):

                # simulate data
                R, y = sample.data_set(base_classifiers_auc, corr, PREVALENCE, N)

                C = utils.compute_mean_cond_corr_matrix(R, y)
                C_upper_tri_vals = utils.extra_upper_diagonal(C)
                corr_reps.append(np.mean(C_upper_tri_vals))

                # train classifier and evaluate performance
                for key, cl in classifiers.items():
                    try:
                        cl.fit(R, y)
                    except NotImplementedError as err:
                        # WOC classifier does not require any training and consequently,
                        # a call to Woc.fit() results in a NotImplementedError.  Safe
                        # to ignore this exception for WOC classifier alone.
                        if key != "WOC":
                            raise err

                    auc = stats.rank_2_auc(cl.compute_ranks(R), y)
                    clreps[key].append(auc)


            # Update conditional correlation value statistics
            corr_stats.update(corr_reps, (i, j))

            # For each classifier, compute mean and sem of auc 
            # over replicate experiments and store
            for key, wstats in clstats.items():
                wstats.update(clreps[key], (i, j))


    # With data collected, make plots

    # Given different conditoinal correlations, plot AUC vs. M base classifiers
    for i, corr in enumerate(CORRELATIONS):

        # Store classifier plot data for each classifier
        for key, wstats in clstats.items():
            clplots[key].update_data(x=M, 
                                    y=wstats.mean[i, :],
                                    yerr=wstats.sem[i,:])

        savename = path.join(DIRECTORY, "auc_100xcorr_{}.pdf".format(int(100*corr)))
        ylabel = "AUC(M, {} = {:0.2f})".format(r"$\hat{r}$", 
                                            np.mean(corr_stats.mean[i, :]))

        plot_errorbars(clplots, 
                       xlabel="Number of Base Classifiers (M)",
                       ylabel= ylabel,
                       savename=savename)

    # Given different number of base classifiers, plot AUC vs. conditional correlation
    for j, m in enumerate(M):

        # Store classifier plot data for each classifier
        for key, wstats in clstats.items():
            clplots[key].update_data(x=corr_stats.mean[:, j],
                                    xerr=corr_stats.sem[:, j],
                                    y=wstats.mean[:, j],
                                    yerr=wstats.sem[:, j])

        savename = path.join(DIRECTORY, "auc_m_{}.pdf".format(m))

        plot_errorbars(clplots, 
                       xlabel= r"Class Conditioned Correlation ($\hat{r}$)",
                       ylabel= "AUC({}, M = {})".format(r"$\hat{r}$", m),
                       savename=savename)

    return 0


if __name__ == "__main__":
    main()

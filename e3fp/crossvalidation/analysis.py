"""Methods for analyzing/summarizing results of cross-validation

Author: Seth Axen
E-mail: seth.axen@gmail.com"""
from itertools import cycle
import logging

import numpy as np
from matplotlib import pyplot as plt

from python_utilities.plotting import KELLY_COLORS, LINESTYLES


def calculate_mean_fp_tp(fp_tp_dicts_iter, num_points=100000):
    """Given dict of roc fp/tp rates, interpolate to average roc curve."""
    mean_tp = 0.0
    mean_fp = np.linspace(0, 1, num_points)

    logging.info("Calculating average ROC with %d points." % num_points)
    target_num = 0
    for i, fp_tp_dict in enumerate(fp_tp_dicts_iter):
        for target_id, fp_tp_rates in fp_tp_dict.iteritems():
            fp = fp_tp_rates[0, :]
            tp = fp_tp_rates[1, :]
            mean_tp += np.interp(mean_fp, fp, tp)
            mean_tp[0] = 0.0
            target_num += 1
        logging.debug("Added ROC curve %d." % (i + 1))

    mean_tp /= target_num
    mean_tp[-1] = 1.0

    return (mean_fp, mean_tp)


def calculate_mean_auc(aucs_dicts_iter):
    aucs_list_dict = {}
    for aucs_dict in aucs_dicts_iter:
        for target_key, auc in aucs_dict.iteritems():
            aucs_list_dict.setdefault(target_key, []).append(auc)

    aucs_dict = {}
    for target_key, auc_list in aucs_list_dict.iteritems():
        try:
            aucs_dict[target_key] = np.mean(auc_list)
        except:
            aucs_dict[target_key] = 0.0

    return aucs_dict


def plot_auc_scatter(aucs_dictx, aucs_dicty, xlabel="X AUCs", ylabel="Y AUCs",
                     title="", figsize=(5, 5), filename=None, plot=True,
                     ref_line=True, color="k"):
    x, y = zip(*[(v, aucs_dicty[k]) for k, v in aucs_dictx.iteritems()
                 if k in aucs_dicty])

    fig = plt.figure(figsize=figsize, dpi=70, frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    if ref_line:
        ax.plot([0, 1], [0, 1], linewidth=1, color="lightgrey", linestyle="--")
    ax.scatter(x, y, s=30, marker="o", facecolors='none', edgecolors=color)
    min_val = min(min(x), min(y))
    ax_min = int(min_val * 4) / 4.0
    ax.set_xlim(ax_min, 1.0)
    ax.set_ylim(ax_min, 1.0)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()

    if not plot:
        return fig
    if filename is not None:
        fig.savefig(filename, dpi=300)
    else:
        plt.show()


def plot_roc_curve(fp_tp_tuples_list, names=None, figsize=(5, 5),
                   filename=None, plot=True, title="", ref_line=True,
                   colors=KELLY_COLORS, linestyles=LINESTYLES):
    if names is not None and len(names) != len(fp_tp_tuples_list):
        names = ["" for i in xrange(len(fp_tp_tuples_list))]

    linestyles_cycle = cycle(LINESTYLES)

    fig = plt.figure(figsize=figsize, dpi=70, frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_color_cycle(KELLY_COLORS.values())
    if ref_line:
        ax.plot([0, 1], [0, 1], linewidth=1, color="lightgrey", linestyle="--")
    for i, fp_tp_tuple in enumerate(fp_tp_tuples_list):
        ax.plot(*fp_tp_tuple, linewidth=2, label=names[i],
                linestyle=linestyles_cycle.next())
    ax.set_xlabel("False Positive Rate (Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(loc=4)
    plt.tight_layout()

    if not plot:
        return fig
    if filename is not None:
        fig.savefig(filename, dpi=300)
    else:
        plt.show()

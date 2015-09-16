"""Functions for creating SEA library and generating background.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import sys
import os
from glob import glob
import logging
import shutil

from ZODB.Connection import ConnectionStateError

from seacore.util.library import Library
from fitcore.fit_main import fit_wizard, suggested_cutoff, round_cutoff,\
                                save_fit, select_only, plot_only, plot_dists,\
                                fit_background
from seashell.cli.library_cli import library_pack, library_inject
from seashell.cli.util import check_is_OK
from seashell.common.adapter.library_adapter import generate_background


RETRY_NUM = 5  # retries when a library building step fails


def fit_wizard_auto(modelfile, library, tc_range, dists=None, testing=False,
                    no_plot=False):
    """Modified from fitcore.fit_main.fit_wizard; auto selects cutoff"""
    if dists is None:
        # having this as an optional argument is useful for debugging
        dists = fit_background(modelfile, tc_range)
    lib_path = library.filename
    lib_dir = os.path.dirname(lib_path)
    lib_name = os.path.splitext(os.path.basename(lib_path))[0]
    dist_file = "{0}-distfit.png".format(lib_name)
    if not testing and not no_plot:
        plotfile = os.path.join(lib_dir, dist_file)
        print "plot dists"
        plot_dists(dists, plotfile)
    suggested = suggested_cutoff(dists)
    print "\nDistribution quality plot shown in browser window."
    print "\tSuggested cutoff from simplistic ratio analysis: %g" % suggested
    if testing:
        cutoff = 0.28
    else:
        print "\tAutomatically picking cutoff as: %g." % suggested
        val = suggested
        val = round_cutoff(val)
        try:
            dists[val]
        except KeyError:
            print "No data for that cutoff. Continuing anyways."
        cutoff = val
    fit = dists[cutoff][1]
    save_fit(library, cutoff, fit)
    print "Fit saved to library file!"


def library_fit(library_file, yes=False, testing=False, plot_prefix=None,
                cutoff=None, no_plot=False):
    """Modified from seashell.cli.library_cli.library_fit"""
    library = Library(library_file)
    try:
        if len(library.fit) > 0:
            if not check_is_OK(
                    "This will overwrite the library's existing fit", yes):
                sys.stderr.write("Aborted.\n")
                sys.exit()
        random_seed = None
        if testing:
            random_seed = 314159
        if cutoff is not None:
            select_only(plot_prefix, cutoff, library)
            library.signature.create()
        else:
            modelfile, tc_range = generate_background(library,
                                                      random_seed=random_seed)
            try:
                if plot_prefix is not None:
                    plot_prefix = os.path.join(plot_prefix)
                    plot_only(modelfile, tc_range, plot_prefix)
                else:
                    if not yes:
                        fit_wizard(modelfile, library, tc_range,
                                   testing=testing)
                    else:
                        fit_wizard_auto(modelfile, library, tc_range,
                                        testing=testing, no_plot=no_plot)

                    try:
                        library.signature.create()
                    except AttributeError:
                        pass

            finally:
                shutil.rmtree(os.path.dirname(modelfile))
    finally:
        library.close()


def build_library(library_file, molecules_file, targets_file, fit_file,
                  library_name=None, generate_fit=False, retry_num=RETRY_NUM,
                  no_plot=True, no_checks=True, log=True):
    """Build library from input files, optionally generating fit."""
    out_fit_file = None
    if generate_fit is True:
        out_fit_file = fit_file
        fit_file = None

    for i in xrange(retry_num):
        try:
            for f in glob("%s*" % library_file):
                os.remove(f)
            library_pack(library_file, molecules_file,
                         targets_file, fit_file,
                         has_fp=True, yes=no_checks)
            if log:
                logging.info("Library built and saved to %s." % library_file)
            break
        except ConnectionStateError:
            logging.warning(
                "Library building failed. Deleting files and restarting (Attempt %d/%d)." % (i+1, retry_num),
                exc_info=True)

    if generate_fit:
        # Generate a fit and add it
        for i in xrange(retry_num):
            try:
                library_fit(library_file, yes=no_checks, no_plot=no_plot)
                if log:
                    logging.info("Fit generated and added to library.")
                break
            except:
                logging.warning(
                    "Adding fit to library failed. Restarting (Attempt %d/%d)." % (i+1, retry_num),
                    exc_info=True)

        if out_fit_file is None:
            out_fit_file = "%s.fit" % os.path.splitext(library_file)[0]

        unpack_fit_from_library(library_file, out_fit_file, log=log)


def inject_fit_into_library(library_file, fit_file):
    """Add fit to existing library."""
    library_inject("fit", library_file, fit_file, yes=True)


def unpack_fit_from_library(library_file, fit_file, log=True):
    """Save fit from existing library"""
    library = Library(library_file)
    try:
        library.fit.write(fit_file)
        if log:
            logging.info("Wrote fit to %s." % fit_file)
    except:
        logging.critical("No fit present in library.")
    finally:
        library.close()

"""Functions for creating SEA library and generating background.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
from glob import glob
import logging

from ZODB.Connection import ConnectionStateError

from seacore.util.library import Library
from seashell.cli.library_cli import library_pack, library_inject, library_fit

RETRY_NUM = 5  # retries when a library building step fails


def build_library(library_file, molecules_file, targets_file, fit_file,
                  library_name=None, generate_fit=False, retry_num=RETRY_NUM,
                  no_plot=True, log=True):
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
                         has_fp=True, no_signature=True,
                         no_checks=True)
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
                library_fit(library_file, no_signature=True, no_checks=True,
                            no_plot=no_plot)
                if log: logging.info("Fit generated and added to library.")
                break
            except:
                logging.warning(
                    "Adding fit to library failed. Restarting (Attempt %d/%d)." % (i+1, retry_num),
                    exc_info=True)

        if out_fit_file is None:
            out_fit_file = "%s.fit" % os.path.splitext(library_file)[0]

        unpack_fit_from_library(library_file, out_fit_file, log=log)


def inject_fit_into_library(library_file, fit_file):
    library_inject("fit", library_file, fit_file, no_signature=True,
                   no_checks=True)


def unpack_fit_from_library(library_file, fit_file, log=True):
    library = Library(library_file)
    try:
        library.fit.write(fit_file)
        if log: logging.info("Wrote fit to %s." % fit_file)
    except:
        logging.critical("No fit present in library.")
    finally:
        library.close()

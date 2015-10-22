"""Utilities for running SEA.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import logging
import shelve

from zc.lockfile import LockError

from seacore.run.core import SEASetCore
from seacore.util.library import TargetKey


class SEASetSearcher(object):

    def __init__(self, reference, log=False, mol_db=None, target_db=None,
                 wipe=True):
        self.log = log
        self.wipe = wipe
        self.core = SEASetCore()
        if mol_db is not None:
            self.set_results_dict = shelve.open(mol_db, writeback=True)
        else:
            self.set_results_dict = {}
        if target_db is not None:
            self.target_results_dict = shelve.open(target_db, writeback=True)
        else:
            self.target_results_dict = {}
        self.open(reference)

    def open(self, reference):
        if self.wipe:
            self.clear_results()
        if self.core.library is not None:
            logging.error(
                "Reference library for search is already defined. Closing.",
                exc_info=True)
            self.close()
        if self.log:
            logging.info("Loading library for searching.")
        self.core.load_reference(reference)
        if self.log:
            logging.info("Library loaded.")

    def search(self, mol_name, fp_list, rerun=False):
        if not rerun and mol_name in self.set_results_dict:
            return self.set_results_dict[mol_name]
        try:
            results = self.core.run(fp_list)
        except AttributeError:
            logging.error(
                "Reference library for search has not been defined.",
                exc_info=True)
            return {}

        logging.debug("SEA Search Results: \n"+repr(results.results))
        # self.set_results_dict[mol_name] = {
        #     target_id: (evalue, max_tc)
        #     for _, target_id, hit_affinity, evalue, max_tc
        #     in results.results}
        for r in results.results:
            _, target_id, hit_affinity, evalue, max_tc = r
            target_key = TargetKey(target_id, hit_affinity)
            self.set_results_dict.setdefault(
                mol_name, {})[target_key] = (evalue, max_tc)
            self.target_results_dict.setdefault(target_key,
                                                set([])).add(mol_name)

        try:
            self.set_results_dict.sync()
        except AttributeError:
            pass
        try:
            self.target_results_dict.sync()
        except AttributeError:
            pass
        return self.set_results_dict.get(mol_name, {})

    def batch_search(self, molecules_lists_dict):
        if self.core.library is None:
            logging.error(
                "Reference library for search has not been defined.",
                exc_info=True)
            return False
        logging.info("Searching sets against library.")
        search_iter = (self.search(k, v)
                       for k, v in molecules_lists_dict.iteritems())
        i = 0
        for result in search_iter:
            if (i + 1) % 100 == 0:
                print("Searched %d molecules." % (i+1))
            i += 1
        if self.log:
            logging.info("Searched %d molecule sets against library." % (i+1))
        return True

    def mol_result(self, mol_name):
        return self.set_results_dict.get(mol_name, {})

    def target_result(self, target_id):
        return self.target_results_dict.get(target_id, set([]))

    def mol_target_result(self, mol_name, target_id):
        return self.set_results_dict.get(mol_name, {}).get(target_id, None)

    def close(self):
        if self.log:
            logging.info("Closing library.")
        self.core.close()

    def clear_results(self):
        if self.log:
            logging.info("Clearing results from searcher.")
        self.set_results_dict.clear()
        self.target_results_dict.clear()

    def __del__(self):
        self.close()


def sea_set_search(library_file, molecules_lists_dict, log=True):
    """Get SEASetSearcher for molecule lists against a SEA library."""
    try:
        searcher = SEASetSearcher(library_file, log=log)
    except LockError:
        os.remove("%s.lock" % library_file)
        searcher = SEASetSearcher(library_file, log=log)

    searcher.batch_search(molecules_lists_dict)
    searcher.close()
    return searcher

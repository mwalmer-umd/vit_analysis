# Helper tools for managing and caching result files
import os
import numpy as np



'''
Given a model ID and a list of analysis methods (strings) to look for, will
check the cache and load any results it finds. It will also return
two lists 'found' and 'not_found' reflecting which analysis methods
were loaded from the cache. If return_dict=True, "results" will be returned
as a dict mapping entries of 'found' to numpy arrays.

blk - Optional parameter for metrics that are computed and cached on a per-block
basis. By default blk=None, for cases when either: 
    * The metric is at the whole-network-level
    * The results for all blocks are stacked together and cached as a single array
'''
def read_results_cache(mod_id, analysis_methods, blk=None, cache_dir='all_results', return_dict=False):
    # check inputs
    if not isinstance(analysis_methods, list):
        analysis_methods = [analysis_methods]
    # read cache
    results = []
    found = []
    not_found = []
    for a_m in analysis_methods:
        if blk is not None:
            fname = os.path.join(cache_dir, mod_id, "%s_blk%02i.npy"%(a_m, blk))
        else:
            fname = os.path.join(cache_dir, mod_id, "%s.npy"%(a_m))
        if os.path.isfile(fname):
            results.append(np.load(fname))
            found.append(a_m)
        else:
            not_found.append(a_m)
    if not return_dict:
        return results, found, not_found
    res_d = {}
    for i in range(len(found)):
        res_d[found[i]] = results[i]
    return res_d, found, not_found



'''
Writes the given results to the correct cache location. "results" may be:
    * A single numpy array if a single result is being cached
    * A list of numpy arrays with the same order as "analysis_methods"
    * A dictionary mapping the entries of "analysis_methods" to numpy arrays
'''
def save_results_cache(results, mod_id, analysis_methods, blk=None, cache_dir='all_results'):
    # check inputs
    if not isinstance(analysis_methods, list):
        analysis_methods = [analysis_methods]
    if not isinstance(results, list) and not isinstance(results, dict):
        results = [results]
    assert len(results) == len(analysis_methods)
    # write cache
    cache_dir_full = os.path.join(cache_dir, mod_id)
    os.makedirs(cache_dir_full, exist_ok=True)
    for i in range(len(results)):
        a_m = analysis_methods[i]
        if isinstance(results, dict):
            res = results[a_m]
        else:
            res = results[i]
        if blk is not None:
            fname = os.path.join(cache_dir, mod_id, "%s_blk%02i.npy"%(a_m, blk))
        else:
            fname = os.path.join(cache_dir, mod_id, "%s.npy"%(a_m))
        np.save(fname, res)
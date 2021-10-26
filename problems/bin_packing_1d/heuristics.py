from problems.bin_packing_1d.definitions import *

"""
    Well known heuristics to solve 1D bin packing
    Used to establis a baseline
"""

# First fit
def first_fit(items, bin_size):
    if len(items) == 0:
        return []

    n_bins = 1
    bins = [Bin(0, bin_size)]

    for item in items:
        #Look for the first bin
        inserted = False
        bin_n = 0
        while not inserted and (bin_n < n_bins):
            if bins[bin_n].free_space >= item.weight:
                bins[bin_n].insert(item)
                inserted = True
            bin_n += 1

        if bin_n == n_bins:
            n_bins += 1
            bins.append(Bin(bin_n, bin_size))
            bins[bin_n].insert(item)
    
    return bins

# Best fit
def best_fit(items, bin_size):
    n_bins = 0
    bins = []

    for item in items:
        best_bin_free_space = bin_size
        best_bin = -1

        #Look for the best bin
        for bin in bins:
            if bin.free_space >= item.weight:
                if bin.free_space < best_bin_free_space:
                    best_bin_free_space = bin.free_space
                    best_bin = bin.id
        
        if best_bin == -1:
            n_bins += 1
            bins.append(Bin(n_bins - 1, bin_size))
            best_bin = n_bins - 1

        bins[best_bin].insert(item)
    
    return bins


# Worst fit
def worst_fit(items, bin_size):
    n_bins = 0
    bins = []

    for item in items:
        best_bin_free_space = -1
        best_bin = -1

        #Look for the best bin
        for bin in bins:
            if bin.free_space >= item.weight:
                if bin.free_space > best_bin_free_space:
                    best_bin_free_space = bin.free_space
                    best_bin = bin.id
        
        if best_bin == -1:
            n_bins += 1
            bins.append(Bin(n_bins - 1, bin_size))
            best_bin = n_bins - 1

        bins[best_bin].insert(item)
    
    return bins

def first_fit_decreasing(items, bin_size):
    sorted_items = sorted(items, key=attrgetter('weight'), reverse=True)
    return first_fit(sorted_items)


def best_fit_decreasing(items, bin_size):
    sorted_items = sorted(items, key=attrgetter('weight'), reverse=True)
    return best_fit(sorted_items)

def worst_fit_decreasing(items, bin_size):
    sorted_items = sorted(items, key=attrgetter('weight'), reverse=True)
    return worst_fit(sorted_items)
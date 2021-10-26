from collections import namedtuple
from problems.common import IState

import math

import numpy as np

"""
    Definitions related to the 1D bin packing problem
"""

class Bin:
    """
        Represents a bin and its state (maximum capacity, occupation, etc.)
    """
    def __init__(self, id, size):
        self.id = id
        self.size = size
        self.items = []
        self.free_space = size
    
    def insert(self, item):
        self.items.append(item)
        self.free_space -= item.weight

    def show(self):
        print('Bin ID: ', self.id, '\t Free space: ', self.free_space)
        print('Items:')
        for item in self.items:
            print(item)
    
    # using copy.deepcopy was too slow => need to do it fast
    def copy(self):
        new_bin = Bin(self.id, self.size)
        new_bin.items = self.items[:]
        new_bin.free_space = self.free_space
        return new_bin

"""
    Item objects inside bins (no need to create a class)
"""
Item = namedtuple('Item', ['id', 'weight'])


class Action:
    """
        Corresponds to putting an item in a bin
    """
    def __init__(self, item, bin_id):
        self.item = item
        self.bin_id = bin_id
        self.code = str(item.id) + '-' + str(bin_id)
        

    def to_int(self, n_items):
        return self.item.id * n_items + self.bin_id


class State(IState):
    """
        implementing State methods to interact with MC algorithms
    """

    def __init__(self, bins, bin_size, remaining_items, n_items, policy=None):
        self.bins = bins
        self.remaining_items = remaining_items
        self.hashed_state = self.get_hash()
        self.bin_size = bin_size
        self.n_items = n_items

    # similar solutions with different order must have the same hash
    def get_hash(self):
        items = [frozenset([item.id for item in bin.items]) for bin in self.bins]
        return hash(frozenset(items))

    def get_successors(self):   # Probably to be removed
        successors = [] 
        for item in self.remaining_items:
            for bin in self.bins:
                if bin.free_space >= item.weight:
                    new_bins = [b.copy() for b in self.bins]
                    new_remaining_items = self.remaining_items.copy()

                    new_bins[bin.id].insert(item)
                    new_remaining_items.remove(item)
                    successors.append(State(new_bins, new_remaining_items))
            
            new_bins = [b.copy() for b in self.bins]
            new_remaining_items = self.remaining_items.copy()

            new_bin_id = len(new_bins)
            new_bins.append(Bin(new_bin_id, self.bin_size))
            new_bins[new_bin_id].insert(item)

            new_remaining_items.remove(item)
            successors.append(State(new_bins, new_remaining_items))
        return successors

    # get successor State after applying an action
    def get_successor(self, action):
        new_bins = [b.copy() for b in self.bins]
        new_remaining_items = self.remaining_items.copy()

        if action.bin_id == len(new_bins): #need to create a new bin
            new_bins.append(Bin(action.bin_id, self.bin_size))

        new_bins[action.bin_id].insert(action.item)
        new_remaining_items.remove(action.item)

        return State(new_bins, self.bin_size, new_remaining_items, self.n_items)

    # get "leagal" actions in a State (e.g., can't put an object in a full bin)
    def get_actions(self):
        actions = []
        for item in self.remaining_items:
            for bin in self.bins:
                if bin.free_space >= item.weight:
                    actions.append(Action(item, bin.id))
            
            actions.append(Action(item, len(self.bins)))
        return actions

    def is_over(self):
        return len(self.remaining_items) == 0
    
    def get_score(self):
        return - len(self.bins)

    
    # encode a state as a matrix (similar solutions must have the same representation)
    def get_representation_matrix(self):
        representation_matrix = np.zeros((self.n_items, self.bin_size, 2))

        sorted_bins = sorted(self.bins, key=lambda bin: bin.free_space)
        for i, bin in enumerate(sorted_bins):
            occupied_space_bin = sum([item.weight for item in bin.items])
            for j in range(occupied_space_bin):
                representation_matrix[i][j][0] = 1

        sorted_remaining_items = sorted(self.remaining_items, key=lambda item: item.weight, reverse=True)
        for i, item in enumerate(sorted_remaining_items):
            for j in range(item.weight):
                representation_matrix[i][j][1] = 1

        return representation_matrix.reshape(1, self.n_items, self.bin_size, 2)


"""
    Helper functions
"""

# returns the action which took us from the old bin list to the new one
def get_move(old_solution_bins, new_solution_bins):
    if len(old_solution_bins) < len(new_solution_bins):
        return Action(new_solution_bins[-1].items[0], new_solution_bins[-1].id)
    
    for old_bin, new_bin in zip(old_solution_bins, new_solution_bins):
        if old_bin.free_space > new_bin.free_space:
            return Action(new_bin.items[-1], new_bin.id)


# printing a solution
def show_solution(items, solution, bin_size, show_bins=False):
    print('Items: ', items)
    print('Trivial lower bound: ', math.ceil(sum([item.weight for item in items]) / bin_size))
    print('Solution length: ', len(solution))
    if show_bins:
        for bin in solution:
            print('--------')
            bin.show()
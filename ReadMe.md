# Context
Monte Carlo methods have shown to be very effective in problems involving exploring a search space (e.g. optimization problems, zero sum games, etc.) This project was realized for the Monte Carlo Methods course in Paris Dauphine - PSL University. I chose to tackle a classical optimization problem (Bin packing) using various MC methods.

# Bin Packing Problem
Informally, the Bin Packing Problem (BPP) can be formulated as follows: one has n items, each item i has a weight w_i, the objective is to pack all items in bins of capacity c such that the number of bins is minimal. This problem is NP-Hard.

# Implemented algorithms

## Heuristics
The BPP has been thoroughly studied in literature. Very popular resolution approaches are heuristic methods. They present the advantage of being simple, efficient and very quick. We implemented the following algorithms:

* First Fit: at each step, put the item i in the first bin where the free space ≥ w_i 
* Best Fit: at each step, put the item i in the bin where the free space is closest to w_i 
* Worst Fit: at each step, put the item i in the bin where the free space is furthest from w_i 
We also implement variants of these algorithms where we first order items in decreasing order of their size (First/Best/Worst Fit Decreasing).


## Monte Carlo Tree Search
Although not always suitable for optimization problems, we implemented MCTS for sake of comparison

## Nested methods
* Nested Monte Carlo Search (NMCS): proposed by Cazenave(2009), it looks for the best solution by leveraging nested levels of random playouts to guide the search.
* Nested Rollout Policy Adaptation (NRPA): proposed by Rosin(2011), same principle as NMCS (uses nested levels) + policy learning. The algorithm adapts the weights of the moves according to the best sequence of moves found so far.

## Ranked Rewards (R2)
By Laterre et al.(2018). The authors suggest an alpha-zero-like algorithm for optimization problems. In alpha zero, the main idea is that an agent should outperform itself through iterations (this is achieved with self plays). In optimization problems, the authors propose using ”ranked rewards”: instead of considering obtained rewards to construct the dataset, one can use reshaped rewards (1 if the agent did better than the ith percentile of previous data, -1 otherwise). 

We adapt the solution to the simple 1D Bin Backing Problem (the reward function is −nb bins)

### Problem representation in a Neural Network 
We propose to represent a bin packing state as a
2 × bin_size × n_items matrix. The first plan represents the content of each bin, and the second one
represents the remaining items. E.g.: In the following, each column represents a bin (there are, at most,
n_items bins) and each row represents a weight unit.

1. Initial state: The first matrix is empty. I.e. no object has been assigned to any bin. And the second matrix contains the objects to be assigned (as if each object was assigned to its own bin)

![Initial state](https://github.com/IsraMekki/MC_Bin_Packing/blob/master/imgs/img1.png?raw=true)


2. Towards the end: The first matrix contains assigned items, and the second one contains the last item to be assigned (note that each time an item is assigned, we shift remaining items to the left)


![Towards the end](https://github.com/IsraMekki/MC_Bin_Packing/blob/master/imgs/img2.png?raw=true)

__Some comments:__
* This representation results in loss of generality since it only encompasses problems of size <= n (problems with n items or less)
* In our implementation, we sort the bins (in both planes) by free space, so that the representation is more robust (two equivalent solutions with different orders would have the same representation)
* The number of outputs of the neural network is n_items^2 (worst case, each item is assigned to its own bin) which is a lot, but we could not think about a better representation.

# Organization
This repo is ogranized as follows:
```
|ReadMe.md
|_experiments.ipynb
|_problems
        |_common.py
        |_bin_packing_1d
                       |_definitions.py
                       |_heuristics.py
                       |_Scholl_1
                            |_...(Scholl_1 instances)
                       |_Hard28
                            |_...(Hard28 instances)
|_MCTS.py
|_nested_methods.py
|_ranked_rewards.py
|_utils.py
|_models
        |_ ...(saved models)
```

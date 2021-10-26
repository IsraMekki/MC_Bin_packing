import math
import numpy as np

from utils import copy_dict_of_dict

"""
    Nested Monte Carlo Search (NMCS)
    reccursive function to look for the best solution
"""
def nmcs(state, level, neg_lower_bound):
    best_sequence = []
    highest_score = - math.inf
    visited_nodes = []

    if level == 0:
        seq, score = random_playout(state)
        return seq, score

    while not state.is_over():
        best_sequence_of_level = []
        highest_score_level = - math.inf
        best_move = None

        for move in state.get_actions():
            s = state.get_successor(move)
            s_sequence, s_score = nmcs(s, level - 1, neg_lower_bound)
            if s_score == neg_lower_bound:
                return visited_nodes + [move] + s_sequence, s_score
            if s_score >= highest_score_level:
                highest_score_level = s_score
                best_sequence_of_level = s_sequence[:]
                best_move = move

        if highest_score_level > highest_score:
            visited_nodes.append(best_move)
            best_sequence = visited_nodes[:] + best_sequence_of_level
            highest_score = highest_score_level
        else:
            best_move = best_sequence[len(visited_nodes)]
            visited_nodes.append(best_move)
        state = state.get_successor(best_move)

    return best_sequence, highest_score

"""
    Nested Rollout Policy Adaptation (NRPA)
    improves NMCS by adapting a rollout policy
"""
def nrpa(root, policy, level, neg_lower_bound, n_iterations=100):
    best_sequence = []
    highest_score = - math.inf
    visited_nodes = []

    if root.hashed_state not in policy:
        policy[root.hashed_state] = dict()
        possible_moves = root.get_actions()
        for move in possible_moves:
            policy[root.hashed_state][move.code] = 1/len(possible_moves)

    if level == 0:
        seq, score = random_playout(root, policy)
        return seq, score
    
    for i in range(n_iterations):
        sequence, score = nrpa(root, policy, level - 1, neg_lower_bound, n_iterations)

        if score >= highest_score:
            highest_score = score
            best_sequence = sequence[:]

        policy = adapt_policy(root, policy, best_sequence)

    return best_sequence, highest_score
            
# used by NRPA
def adapt_policy(root, policy, sequence, alpha=1.):
    state = root
    new_policy = copy_dict_of_dict(policy)
    for move in sequence:
        new_policy[state.hashed_state][move.code] += alpha

        possible_moves = state.get_actions()
        z = sum([np.exp(policy[state.hashed_state][move.code]) for move in possible_moves])

        for possible_move in possible_moves:
            new_policy[state.hashed_state][possible_move.code] -= alpha * np.exp(policy[state.hashed_state][possible_move.code]) / z

        state = state.get_successor(move)

    policy = new_policy
    return policy

            
# can be made epsilon greedy
def random_playout(state, policy=None):
    rollout_state = state
    actions_list = []
    while not rollout_state.is_over():
        possible_moves = rollout_state.get_actions()
        if policy is None:
            move = possible_moves[np.random.randint(len(possible_moves))]

        else:
            if rollout_state.hashed_state not in policy:
                policy[rollout_state.hashed_state] = dict()
                possible_moves = rollout_state.get_actions()
                for move in possible_moves:
                    policy[rollout_state.hashed_state][move.code] = 1/len(possible_moves)

            z = sum([np.exp(policy[rollout_state.hashed_state][move.code]) for move in possible_moves])
            weights = [np.exp(policy[rollout_state.hashed_state][move.code]) / z for move in possible_moves]
            move = np.random.choice(possible_moves, p=weights)

        rollout_state = rollout_state.get_successor(move)
        actions_list.append(move)
    return actions_list, rollout_state.get_score()


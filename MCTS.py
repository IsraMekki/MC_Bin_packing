import numpy as np

"""
    Monte Carlo Tree Search implementation
"""

class Node(object):
    """
        Node representation in MCTS
        each node contains the state of the game as well as useful statistics
    """
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.successors = []
        self.n_visits = 0
        self.results = []
        self.untried_moves = self.state.get_actions()


    def value(self):
        if len(self.results) == 0:
            return 0
        return np.mean(self.results)


    def expand(self, transposition_table=None):
        move = self.untried_moves.pop()
        successor_state = self.state.get_successor(move)
        
        successor_node = Node(state=successor_state, parent=self)
        self.successors.append(successor_node)

        if transposition_table is not None:
            h = successor_state.hashed_state
            if h in transposition_table:        
                successor_node.h = h
                successor_node.n_visits = transposition_table[h]['visits']
                successor_node.results = transposition_table[h]['results']
                return None
            else:
                transposition_table[h] = {
                    'visits': 0,
                    'results': []
                }
        return successor_node

    def is_terminal(self):
        return self.state.is_over()

    def random_action(self, possible_actions):        
        return possible_actions[np.random.randint(len(possible_actions))]

    # interesting to make it epsilon greedy (using heuristics for example)
    def random_rollout(self):
        rollout_state = self.state
        while not rollout_state.is_over():
            possible_actions = rollout_state.get_actions()
            action = self.random_action(possible_actions)

            rollout_state = rollout_state.get_successor(action)
        return rollout_state.get_score()


    def back_propagate(self, score, transposition_table=None):
        self.n_visits += 1
        #if self.state.is_over():
        self.results.append(score)

        if transposition_table is not None:
            h = self.state.hashed_state
            if h in transposition_table:
                transposition_table[h]['visits'] += 1
                if self.state.is_over():
                    transposition_table[h]['results'].append(score)
        if self.parent:
            self.parent.back_propagate(score, transposition_table)

    def all_successors_visited(self):
        return len(self.untried_moves) == 0

    def select_successor(self, c=1.4, transposition_table=None, policy=None):
        self_visits = 0
        for s in self.successors:
            if transposition_table is not None:
                h = s.state.hashed_state
                if h in transposition_table:
                    if transposition_table[h]['visits'] > s.n_visits:
                        s.n_visits = transposition_table[h]['visits']
                        s.results = transposition_table[h]['results']
            self_visits += s.n_visits

        if policy is None:
            uct_scores = [(s.value() / s.n_visits + c * np.sqrt((2 * np.log(self_visits) / s.n_visits))) for s in self.successors]
        else:
            uct_scores = [(s.value() / s.n_visits + c * p * np.sqrt((2 * np.log(self_visits) / s.n_visits))) for s, p in zip(self.successors, policy)]
        return self.successors[np.argmax(uct_scores)]




class MCTS:
    """
        Getting the best successor by performing Monte Caro Tree Search
        - Supports the of transposition tables
        - Supports the use of a neural network (e.g., alpha zero)
    """
    def __init__(self, node, transposition_table=None, nn=None):
        self.root = node
        self.transposition_table = transposition_table
        self.nn = nn


    def best_successor(self, n_simulations):
        for i in range(n_simulations):   
            v = self.selection_policy()
            policy_legal_actions = None
            if self.nn == None or v.is_terminal():
                reward = v.random_rollout()
            else:
                policy, reward = self.nn.predict(v.state.get_representation_matrix())
                policy = policy[0]
                reward = -1 if reward[0][0] < 0 else 1
                
                possible_actions = v.state.get_actions()
                possible_actions = [action.to_int(v.state.n_items) for action in possible_actions]
                policy_legal_actions = [policy[i] for i in range(v.state.n_items**2) if i in possible_actions]

            
            v.back_propagate(reward, transposition_table=self.transposition_table)

        return self.root.select_successor(c=1., transposition_table=self.transposition_table, policy=policy_legal_actions)


    def selection_policy(self):
        current_node = self.root
        while not current_node.is_terminal():
            while not current_node.all_successors_visited():
                    expanded_node = current_node.expand(self.transposition_table)
                    if expanded_node:
                        return expanded_node

            current_node = current_node.select_successor(transposition_table=self.transposition_table)
        return current_node
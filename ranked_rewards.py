
import numpy as np

from problems.bin_packing_1d.definitions import *
from MCTS import *

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

class R2:
    """
        solving 1-player games with Ranked Rewards (R2)
    """

    def __init__(self, nn, percentile=75, mini_batch_size=32, n_iterations=10, 
                n_simulations=20, n_episodes=10, epochs=10, buffer_size=64, dataset_size=128):
        self.percentile = percentile
        self.mini_batch_size = mini_batch_size
        self.nn = nn
        self.n_iterations = n_iterations
        self.n_simulations = n_simulations
        self.n_episodes = n_episodes
        self.epochs = epochs
        self.buffer_size = buffer_size
        self.dataset_size = dataset_size

    def r2(self, dataset, model_dir, bin_size, n_items):
        data = []
        buffer = []
        for i in range(self.n_iterations):
            for episode in range(self.n_episodes):
                items = dataset[np.random.randint(len(dataset))]
                state_0 = State([], bin_size, items, n_items)
                node_0 = Node(state_0)
                moves = []
                states = [state_0]

                mcts_solver = MCTS(node_0, nn=self.nn)
                next_state = mcts_solver.best_successor(self.n_simulations)
                
                moves.append(get_move(node_0.state.bins, next_state.state.bins))
        
                while not next_state.is_terminal():
                    mcts_solver = MCTS(next_state, nn=self.nn)
                    new_next_state = mcts_solver.best_successor(self.n_simulations)
                    
                    states.append(next_state.state)
                    moves.append(get_move(next_state.state.bins, new_next_state.state.bins))
                    
                    next_state = new_next_state
                
                reward = next_state.state.get_score()

                if len(buffer) < self.buffer_size:
                    buffer.append(reward)
                else:
                    buffer[1:] += [reward]
                    
                thr = np.percentile(buffer, self.percentile)
                z = self.reshape_reward(reward, thr)

                for state, move in zip(states, moves):
                    if len(data) < self.dataset_size :
                        data.append([state.get_representation_matrix()[0], move.to_int(len(items)), z])
                    else:
                        data[1:] += [[state.get_representation_matrix()[0], move.to_int(len(items)), z]]

            
            
            split_ind = int(len(data) * .8)
            train_data = data[:split_ind]
            val_data = data[split_ind:]

            train_generator = self.data_generator(train_data)
            val_generator = self.data_generator(val_data)


            checkpoint = ModelCheckpoint(model_dir + '/model_' + str(i) + '.h5' , monitor='val_loss', verbose=1, save_best_only=True, mode='min') 

            history = self.nn.fit_generator(
                train_generator,
                steps_per_epoch=1,
                epochs=self.epochs,
                validation_data=val_generator,
                validation_steps=len(val_data) // self.mini_batch_size, 
                callbacks=[checkpoint])


                
    def reshape_reward(self, reward, thr):
        if reward > thr:
            return 1

        if reward < thr:
            return -1

        if reward == thr:
            v = np.random.binomial(1, .5, 1)[0]
            if v == 1:
                return 1
            else:
                return -1


    def data_generator(self, data):
        indices = list(range(len(data)))
        while True:
            np.random.shuffle(indices)

            for offset in range(0, len(data), self.mini_batch_size):
                
                batch_indices = indices[offset:offset+self.mini_batch_size]

                X = []
                y = {'policy': [], 'value': []}

                for batch_index in batch_indices:
                    
                    X.append(data[batch_index][0])
                    y['policy'].append(data[batch_index][1])
                    y['value'].append(data[batch_index][2])

                X = np.array(X)
                y['policy'] = np.array(y['policy'])
                y['value'] = np.array(y['value'])

        
                yield X, y
from abc import ABC, abstractmethod


class IState(ABC):
    """
        Methods that need to be implemented to use Monte Carlo methods
    """
    @abstractmethod
    def get_hash(self):
        pass

    @abstractmethod
    def get_successor(self, action):
        pass

    @abstractmethod
    def get_actions(self):
        pass

    @abstractmethod
    def is_over(self):
        pass
    
    @abstractmethod
    def get_score(self):
        pass

    @abstractmethod
    def get_representation_matrix(self):
        pass
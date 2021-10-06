from abc import ABC, abstractmethod

class ABCModel(ABC):
    """Abstract Base Class Definition"""

    @abstractmethod
    def load_model(self):
        """Required method"""

    @abstractmethod
    def get_data(self):
        """Required method"""
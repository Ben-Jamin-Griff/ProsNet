from abc import ABC, abstractmethod

class ABCModel(ABC):
    """Abstract Base Class Definition"""

    @abstractmethod
    def load_model(self):
        """Required method"""

    @abstractmethod
    def get_data(self):
        """Required method"""

    #@abstractmethod
    #def predict_postures(self):
        """Required method"""

    #@abstractmethod
    #def save_results(self):
        """Required method"""
from abc import ABC, abstractmethod

class ABCDataset(ABC):
    """Abstract Base Class Definition"""

    @abstractmethod
    def get_data(self):
        """Required method"""

    @abstractmethod
    def get_posture_stack(self):
        """Required method"""

    @abstractmethod
    def show_set(self):
        """Required method"""

    @abstractmethod
    def save_set(self):
        """Required method"""
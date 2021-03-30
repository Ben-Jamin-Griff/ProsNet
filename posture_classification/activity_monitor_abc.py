from abc import ABC, abstractmethod

class ABCActivityMonitor(ABC):
    """Abstract Base Class Definition"""

    @abstractmethod
    def load_raw_data(self):
        """Required method"""

    @abstractmethod
    def save(self, output, data):
        """Required method"""
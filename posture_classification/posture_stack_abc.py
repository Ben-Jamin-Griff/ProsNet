from abc import ABC, abstractmethod

class ABCPostureStack(ABC):
    """Abstract Base Class Definition"""

    @abstractmethod
    def get_data(self):
        """Required method"""

    @abstractmethod
    def create_stack(self):
        """Required method"""

#    @abstractproperty
#    def posture_stack(self):
#        """Required property"""
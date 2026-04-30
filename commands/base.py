from abc import ABC, abstractmethod

class Command(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def execute(self):
        pass
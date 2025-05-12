from abc import ABC, abstractmethod

class ModelInterface(ABC):
    
    def __init__(self):
        print("Creating new model...")
        self.model = None
    
    @abstractmethod
    def load(self, path:str):
        pass

    @abstractmethod
    def infer(self, prompt: str, duration: float, steps:int):
        pass
    
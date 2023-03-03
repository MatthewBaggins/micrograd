import random
from micrograd.value import Value

class Neuron:
    def __init__(self, n_in: int) -> None:
        """A neural network neuron
        
        Args
        ----------
        n_in: int
            - size of previous layer
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1, 1))
        
    def __call__(self, xs: list[Value]) -> Value:
        return sum((w * x for w, x in zip(self.w, xs)), self.b).tanh() #type:ignore

    @property
    def parameters(self) -> list[Value]:
        return [*self.w, self.b]

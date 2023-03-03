from micrograd.neuron import Neuron
from micrograd.value import Value


class Layer:
    """Layer of many `Neuron`s"""

    def __init__(self, d_in: int, d_out: int) -> None:
        """Args
        ----------
        d_in: int
            - number of neurons in previous layer
            and thus also number of params in each neuron in this layer
            (if this is not the input layer)
        d_out: int
            - number of neurons in this layer
            and thus also number of this layer's outputs and number of neurons in the next layer
            (if this is not the final layer)
        """
        self.neurons = [Neuron(d_in) for _ in range(d_out)]

    def __call__(self, xs: list[Value]) -> list[Value]:
        return [n(xs) for n in self.neurons]

    @property
    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters]

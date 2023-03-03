from typing import cast

from micrograd.layer import Layer
from micrograd.value import Value


class MLP:
    """Simple [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)"""
    def __init__(self, d_in: int, d_hidden: int, n_hidden: int, d_out: int) -> None:
        """Args
        ----------
        d_in: int
            - number of values in the input
        d_hidden: int
            - number of neurons in each hidden layer
        n_hidden: int
            - number of hidden layers, each with `d_hidden` neurons
        d_out: int
            - number of values in the output
        """
        self.layer_in = Layer(d_in=d_in, d_out=d_hidden)
        self.layers_hidden = [
            Layer(d_in=d_hidden, d_out=d_hidden) for _ in range(n_hidden)
        ]
        self.layer_out = Layer(d_in=d_hidden, d_out=d_out)

    def __call__(self, xs: list[Value]) -> list[Value]:
        for layer in self.layers:
            xs = layer(xs)
        return xs

    def batch_call(self, xs: list[list[Value]]) -> list[list[Value]]:
        """Call this network on many batches"""
        return [self(x) for x in xs]

    @property
    def layers(self) -> list[Layer]:
        return [self.layer_in, *self.layers_hidden, self.layer_out]

    @property
    def parameters(self) -> list[Value]:
        return [p for l in self.layers for p in l.parameters]

    @staticmethod
    def compute_loss(ys: list[float], preds: list[Value]) -> Value:
        """Compute loss, given true values (`ys`)
        and predicted (by this network) values `preds`
        """
        return cast(Value, sum((y - pred) ** 2 for y, pred in zip(ys, preds)))

    def step(self, loss: Value, lr: float = 1e-4) -> None:
        """Do one backpropagation step, based on `loss` and learning rate (`lr`)"""
        loss.backprop()
        for p in self.parameters:
            assert p.grad
            p.val -= lr * p.grad

    def zero_grad(self) -> None:
        """Set gradients of all parameters to zero"""
        for p in self.parameters:
            p.grad = 0

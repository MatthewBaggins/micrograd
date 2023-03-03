from __future__ import annotations

from functools import reduce
from math import prod, tanh, exp
from operator import or_
from typing import Callable, Literal

from typing_extensions import Self

OP = Literal["add", "mul", "pow", "abs", "sigmoid", "tanh", "relu", "exp"]
"""Operations supported by value"""


def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


def relu(x: float) -> float:
    return max(0.0, x)


class Value:
    """A value type that supports storing computational graph, derivatives, backprop etc."""

    __slots__ = ("val", "label", "grad", "children", "op", "backward")
    val: float
    label: str | None
    grad: float
    children: set[Value]
    op: OP | None
    backward: Callable[[], None]

    @classmethod
    def make(cls, *vals: float) -> list[Self]:
        """Alternative constructor: make many `Value`s in one function"""
        return [cls(val) for val in vals]

    def __init__(
        self,
        val: float,
        label: str | None = None,
        *,
        children: set[Value] | None = None,
        op: OP | None = None,
    ) -> None:
        """Args
        ----------
        val: float
            - value
        label: str | None = None
            - optional identifier
        children: set[Value] | None = None
            - children of that value (used to computed it)
        op: OP | None = None
            - identifies operation used to computed that value
        """
        self.val = val
        self.label = label
        self.grad = 0.0 # gradient
        self.children = set(children) if children is not None else set()
        self.op = op
        self.backward = lambda: None # default backward gradient computation operation

    ##############
    #    Math    #
    ##############

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.val + other.val, children={self, other}, op="add")

        def backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out.backward = backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.val * other.val, children={self, other}, op="mul")

        def backward() -> None:
            self.grad += other.val * out.grad
            other.grad += self.val * out.grad

        out.backward = backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __abs__(self) -> Value:
        out = Value(abs(self.val), children={self}, op="abs")

        def backward() -> None:
            if out.val > 0:
                self.grad = out.grad
            elif out.val < 0:
                self.grad = -out.grad

        out.backward = backward
        return out

    def __pow__(self, other: float) -> Value:
        out = Value(self.val**other, children={self}, op="pow")

        def backward() -> None:
            self.grad += other * self.val ** (other - 1) * out.grad

        out.backward = backward
        return out

    def sigmoid(self) -> Value:
        """[Sigmoid/logistic function](https://en.wikipedia.org/wiki/Logistic_function)"""
        val = sigmoid(self.val)
        out = Value(val, children={self}, op="sigmoid")

        def backward() -> None:
            self.grad = val * (1 - val) * out.grad

        out.backward = backward
        return out

    def tanh(self) -> Value:
        """[Hyperbolic tangent function](https://en.wikipedia.org/wiki/Hyperbolic_functions#Hyperbolic_tangent)"""
        val = tanh(self.val)
        out = Value(val, children={self}, op="tanh")

        def backward() -> None:
            self.grad += (1 - val**2) * out.grad

        out.backward = backward
        return out

    def relu(self) -> Value:
        """[REctified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))"""
        val = relu(self.val)
        out = Value(val, children={self}, op="relu")

        def backward() -> None:
            self.grad += (val > 0) * out.grad

        out.backward = backward
        return out

    def exp(self) -> Value:
        """Exponentiation (e^x)"""
        out = Value(exp(self.val), children={self}, op="exp")

        def backward() -> None:
            self.grad += out.val * out.grad

        out.backward = backward
        return out

    ##################
    #    Backprop    #
    ##################

    def backprop(self) -> None:
        """Set this `Value`'s gradient to 1 and backpropagate to compute its descendants gradients"""
        self.grad = 1.0
        topo = []
        visited = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for c in v.children:
                    build_topo(c)
                topo.append(v)

        build_topo(self)
        for n in reversed(topo):
            n.backward()

    @property
    def grad_dict(self) -> dict[str, float]:
        """Dictionary mapping descendant labels to their gradients"""
        descendants = self.get_descendants()
        return {
            (d.label if d.label else str(i)): d.grad for i, d in enumerate(descendants)
        }

    ########################
    #    Naive backprop    #
    ########################

    def recompute(self, mod: dict[str, float] | None = None) -> float:
        """Recompute what would be this `Value`'s `val` 
        if one or more of its descendants changed its/their values.
        """
        if mod is None or self.label not in mod:
            new_val = self.val
        else:
            new_val = mod[self.label]
        if self.op is None:
            return new_val
        recomputed_children = [c.recompute(mod) for c in self.children]
        match self.op:
            case "add":
                return sum(recomputed_children)
            case "mul":
                return prod(recomputed_children)
            case "abs":
                assert not recomputed_children
                if new_val > 0:
                    return 1
                if new_val < 0:
                    return -1
                return 0
            case "pow":
                assert len(recomputed_children) == 2
                return (recomputed_children[1] - 1) * recomputed_children[0]
            case "sigmoid":
                assert len(recomputed_children) == 1
                return sigmoid(self.val)
            case "tanh":
                assert len(recomputed_children) == 1
                return tanh(self.val)
            case "relu":
                assert len(recomputed_children) == 1
                return relu(self.val)
            case "exp":
                assert len(recomputed_children) == 1
                return exp(self.val)

    def naive_backprop(self, h: float = 1e-5) -> None:
        """Naive backprop. Doesn't work for some reason. 
        Bonus points for you if you find a way to make it work ;3
        """
        self.grad = 1.0
        descendants = self.get_descendants(include_leafs=True)
        descendant_dict = {d.label: d for d in descendants if d.label is not None}
        for label, d in descendant_dict.items():
            mod = {label: d.val + h}
            new_val = self.recompute(mod)
            grad = (new_val - self.val) / h
            descendant_dict[label].grad = grad

    def get_descendants(self, *, include_leafs: bool = False) -> set[Value]:
        """Get all descendants of this `Value`"""
        return reduce(
            or_,
            (
                {c, *c.get_descendants(include_leafs=include_leafs)}
                for c in self.children
            ),
            set(),
        )

    ###################
    #   Other magic   #
    ###################

    def __repr__(self) -> str:
        return (
            f"Value(data={self.val}"
            + (f', label="{self.label}"' if self.label else "")
            + ")"
        )

    def __str__(self) -> str:
        return repr(self)

    def __or__(self, other: str) -> Value:
        """This makes it possible to assign labels with pipe (`|`), hooray.
        
        ```py
        x = Value(1) | "x's label"
        print(x.label) # "x's label"
        ```        
        """
        self.label = other
        return self

"""Basically copy-pasted from Karpathy's code. I only added types."""
from typing import Literal

from graphviz import Digraph

from micrograd.value import Value


def trace(root: Value) -> tuple[set[Value], set[tuple[Value, Value]]]:
    nodes, edges = set(), set()

    def build(v: Value):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(
    root: Value,
    format_: Literal["png", "svg"] = "svg",
    rankdir: Literal["LR", "TB"] = "LR",
) -> Digraph:
    nodes, edges = trace(root)
    dot = Digraph(format=format_, graph_attr={"rankdir": rankdir})
    for n in nodes:
        n_label = f" {n.label } | " if n.label is not None else ""
        dot.node(
            name=str(id(n)),
            label="{" f" {n_label} data {n.val:.4f} | grad {n.grad:.4f}" "}",
            shape="record",
        )
        if n.op:
            dot.node(name=str(id(n)) + n.op, label=n.op)
            dot.edge(str(id(n)) + n.op, str(id(n)))
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + (n2.op or ""))
    return dot

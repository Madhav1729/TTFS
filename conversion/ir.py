"""Graph IR of the conversion: an ordered list of nodes (conv, linear, ttfs_neuron, pool, ...)."""

from dataclasses import dataclass, field

OPS = {
    "conv",
    "linear",
    "matmul",
    "bn",
    "quant",
    "relu",
    "pool_max",
    "pool_avg",
    "add",
    "flatten",
    "layernorm",
    "softmax",
    "ttfs_neuron",
    "input",
    "output",
}


@dataclass
class Node:
    name: str
    op: str
    inputs: list = field(default_factory=list)
    attrs: dict = field(default_factory=dict)
    digital: bool = False  # runs on a digital island, not the spike array

    def __post_init__(self):
        if self.op not in OPS:
            raise ValueError(f"unknown op {self.op!r}")


@dataclass
class Graph:
    nodes: list = field(default_factory=list)

    def add(self, node):
        if any(n.name == node.name for n in self.nodes):
            raise ValueError(f"duplicate node {node.name!r}")
        missing = [i for i in node.inputs if i not in self.names()]
        if missing:
            raise ValueError(f"{node.name}: inputs {missing} not defined yet")
        self.nodes.append(node)
        return node

    def names(self):
        return [n.name for n in self.nodes]

    def __getitem__(self, name):
        return next(n for n in self.nodes if n.name == name)

    def of(self, op):
        return [n for n in self.nodes if n.op == op]

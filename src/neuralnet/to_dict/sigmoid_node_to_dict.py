from typing import TypedDict
from src.neuralnet.sigmoid_node import SigmoidNode


class NodeDict(TypedDict, total=False):
    bias: str

class SigmoidNodeToDict():
    def to_dict(self, node: SigmoidNode) -> NodeDict:
        return {
            'bias': str(node.bias)
        }
        
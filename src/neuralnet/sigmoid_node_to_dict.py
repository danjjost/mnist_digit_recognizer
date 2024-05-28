from typing import TypedDict
from src.neuralnet.sigmoid_node import SigmoidNode


class NodeDict(TypedDict, total=False):
    id: str
    bias: str

class SigmoidNodeToDict():
    def to_dict(self, node: SigmoidNode) -> NodeDict:
        return {
            'id': node.id,
            'bias': str(node.bias)
        }
        
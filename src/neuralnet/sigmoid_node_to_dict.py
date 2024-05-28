from decimal import Decimal
from typing import TypedDict
from src.neuralnet.sigmoid_node import SigmoidNode


class NodeDict(TypedDict, total=False):
    id: str
    bias: Decimal

class SigmoidNodeToDict():
    def to_dict(self, node: SigmoidNode) -> NodeDict:
        return {
            'id': node.id,
            'bias': node.bias
        }
        
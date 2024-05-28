from decimal import Decimal
from typing import TypedDict
from src.neuralnet.synapse import Synapse

class SynapseDict(TypedDict):
    id: str
    weight: Decimal

class SynapseToDict():
    def to_dict(self, synapse: Synapse) -> SynapseDict:
        return {
            'id': synapse.id,
            'weight': synapse.weight
        }
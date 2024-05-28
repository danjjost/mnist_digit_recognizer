from typing import TypedDict
from src.neuralnet.synapse import Synapse

class SynapseDict(TypedDict):
    id: str
    weight: str

class SynapseToDict():
    def to_dict(self, synapse: Synapse) -> SynapseDict:
        return {
            'id': synapse.id,
            'weight': str(synapse.weight)
        }
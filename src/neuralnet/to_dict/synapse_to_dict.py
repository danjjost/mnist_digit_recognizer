from typing import TypedDict
from src.neuralnet.synapse import Synapse

class SynapseDict(TypedDict):
    weight: str

class SynapseToDict():
    def to_dict(self, synapse: Synapse) -> SynapseDict:
        return {
            'weight': str(synapse.weight)
        }
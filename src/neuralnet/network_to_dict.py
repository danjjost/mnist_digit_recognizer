from typing import TypedDict
from src.neuralnet.network import Network
from src.neuralnet.sigmoid_node import SigmoidNode
from src.neuralnet.sigmoid_node_to_dict import NodeDict, SigmoidNodeToDict
from src.neuralnet.synapse import Synapse
from src.neuralnet.synapse_to_dict import SynapseDict, SynapseToDict

class NetworkDictionary(TypedDict):
    id: str
    learning_rate: str
    node_layers: list[list[NodeDict]]
    synapse_layers: list[list[SynapseDict]]


class NetworkToDict():    
    def to_dict(self, network: Network) -> NetworkDictionary:
        node_layers = self.get_node_layers(network)
        synapse_layers = self.get_synapse_layers(network)
    
        return {
            'id': network.id,
            'learning_rate': str(network.learning_rate),
            'node_layers': node_layers,
            'synapse_layers': synapse_layers,
        }

    def get_synapse_layers(self, network: Network) -> list[list[SynapseDict]]:
        dict_synapse_layers: list[list[SynapseDict]] = [] 
        
        for index in range(len(network.synapse_layers)):
            synapse_layer = network.synapse_layers[index]
            dict_synapse_layer: list[SynapseDict] = []
            
            for synapse in synapse_layer:
                synapse_dictionary = SynapseToDict().to_dict(synapse)
                dict_synapse_layer.append(synapse_dictionary)


            dict_synapse_layers.append(dict_synapse_layer)
            
                
        return dict_synapse_layers   
    
    def get_node_layers(self, network: Network) -> list[list[NodeDict]]:
        node_dict_layers: list[list[NodeDict]] = []
        
        for node_layer in network.node_layers:
            node_dict_layer: list[NodeDict] = []
            
                        
            for node in node_layer:
                node_dictionary = SigmoidNodeToDict().to_dict(node)
                node_dict_layer.append(node_dictionary)
            
            
            node_dict_layers.append(node_dict_layer)
                
        return node_dict_layers
    
    def from_dict(self, dictionary: NetworkDictionary) -> Network:
        network_schema = self.get_network_schema(dictionary)
        network = Network(network_schema)
        
        network.id = dictionary.get('id')
        network.learning_rate = float(dictionary.get('learning_rate'))

        self.set_node_layers(network, dictionary)
        self.set_synapse_layers(network, dictionary)        
        
        return network
        
    def get_network_schema(self, dictionary: NetworkDictionary) -> list[int]:
        schema: list[int] = []
        
        for node_layer in dictionary['node_layers']:
            schema.append(len(node_layer))
            
        return schema
    
    
    def set_node_layers(self, network: Network, dictionary: NetworkDictionary):
        for layer_index in range(len(network.node_layers)):
            node_layer = network.node_layers[layer_index]
            node_dict_layer = dictionary['node_layers'][layer_index]
            
            self.set_nodes(node_layer, node_dict_layer)
            
            
    def set_nodes(self, node_layer: list[SigmoidNode], node_dict_layer: list[NodeDict]):
        for node_index in range(len(node_layer)):
            
            node = node_layer[node_index]
            dict_node = node_dict_layer[node_index]
            
            node.id = dict_node.get('id') # type: ignore
            node.bias = float(dict_node.get('bias')) # type: ignore
            
            
    def set_synapse_layers(self, network: Network, dictionary: NetworkDictionary):
        for layer_index in range(len(network.synapse_layers)):
            synapse_layer = network.synapse_layers[layer_index]
            synapse_dict_layer = dictionary['synapse_layers'][layer_index]
            
            self.set_synapses(synapse_layer, synapse_dict_layer)
            
    def set_synapses(self, synapse_layer: list[Synapse], synapse_dict_layer: list[SynapseDict]):
        for synapse_index in range(len(synapse_layer)):
            synapse = synapse_layer[synapse_index]
            dict_synapse = synapse_dict_layer[synapse_index]
            
            synapse.id = dict_synapse.get('id')
            
            synapse.weight = float(dict_synapse.get('weight'))

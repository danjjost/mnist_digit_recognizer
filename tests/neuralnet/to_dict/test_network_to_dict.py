
import json
import unittest

from src.neuralnet.network import Network
from src.neuralnet.to_dict.network_to_dict import NetworkToDict


class TestNetworkToDict(unittest.TestCase):
    def test_network_to_dict(self):
        network = Network([1,2])
        #    / O
        #  O - O
        
        network.node_layers[0][0].bias = float(1.2)
        
        network.synapse_layers[0][0].weight = float(1.3)
        network.synapse_layers[0][1].weight = float(1.4)
        
        network.node_layers[1][0].bias = float(1.5)
        network.node_layers[1][1].bias = float(1.6)
        
        
        dictionaryNetwork = NetworkToDict().to_dict(network)
        
        
        reconstructedNetwork = NetworkToDict().from_dict(dictionaryNetwork)
        
        
        self.assertEqual(reconstructedNetwork.node_layers[0][0].bias, float(1.2))
        
        self.assertEqual(reconstructedNetwork.synapse_layers[0][0].weight, float(1.3))
        self.assertEqual(reconstructedNetwork.synapse_layers[0][1].weight, float(1.4))
        
        self.assertEqual(reconstructedNetwork.node_layers[1][0].bias, float(1.5))
        self.assertEqual(reconstructedNetwork.node_layers[1][1].bias, float(1.6))
        
        
    def test_network_to_schema(self):
        network = Network([5, 4, 2])
     
        network_dictionary = NetworkToDict().to_dict(network)
     
        
        schema = NetworkToDict().get_network_schema(network_dictionary)
        
        
        self.assertEqual(schema, [5, 4, 2])
        
    def test_network_node_references(self):
        network = Network([1,2])
        #    / O
        #  O - O
        
        network.node_layers[0][0].bias = float(1.2)
        
        network.synapse_layers[0][1].weight = float(1.3)

        network.node_layers[1][1].bias = float(1.5)
        
        dictionaryNetwork = NetworkToDict().to_dict(network)
        
        
        reconstructedNetwork = NetworkToDict().from_dict(dictionaryNetwork)
        
        
        second_synapse = reconstructedNetwork.synapse_layers[0][1]
        
        self.assertEqual(second_synapse.weight, float(1.3))
        
        self.assertEqual(second_synapse.input_node.bias, float(1.2))
        
        self.assertEqual(second_synapse.output_node.bias, float(1.5))
        
    def test_dictionary_parsing_to_json(self):
        network = Network([1,2])
        #    / O
        #  O - O
        
        network.node_layers[0][0].bias = float(1.2)
        
        network.synapse_layers[0][0].weight = float(1.3)
        network.synapse_layers[0][1].weight = float(1.4)
        
        network.node_layers[1][0].bias = float(1.5)
        network.node_layers[1][1].bias = float(1.6)
        
        
        dictionaryNetwork = NetworkToDict().to_dict(network)
        
        json_string = json.dumps(dictionaryNetwork)
        dict_from_json = json.loads(json_string)

        
        reconstructedNetwork = NetworkToDict().from_dict(dict_from_json)
        
        
        self.assertEqual(reconstructedNetwork.node_layers[0][0].bias, float(1.2))
        
        self.assertEqual(reconstructedNetwork.synapse_layers[0][0].weight, float(1.3))
        self.assertEqual(reconstructedNetwork.synapse_layers[0][1].weight, float(1.4))
        
        self.assertEqual(reconstructedNetwork.node_layers[1][0].bias, float(1.5))
        self.assertEqual(reconstructedNetwork.node_layers[1][1].bias, float(1.6))
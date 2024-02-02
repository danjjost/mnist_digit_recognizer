import unittest

from src.network import Network

class TestNetwork(unittest.TestCase):
    
    def test_network_builds_with_correct_number_of_layers(self):
        network1 = Network([1, 1, 1, 1])
        network2 = Network([1, 1])
        network3 = Network([1])

        assert len(network1.node_layers) == 4, f"network1 had {len(network1.node_layers)} layers, expected 3"
        assert len(network2.node_layers) == 2, f"network2 had {len(network2.node_layers)} layers, expected 1"
        assert len(network3.node_layers) == 1, f"network3 had {len(network3.node_layers)} layers, expected 0"
        
        
    def test_network_builds_with_correct_number_of_synapse_layers(self):
        network1 = Network([1, 1, 1, 1])
        network2 = Network([1, 1])
        network3 = Network([1])

        assert len(network1.synapse_layers) == 3, f"network1 had {len(network1.synapse_layers)} synapse layers, expected 3"
        assert len(network2.synapse_layers) == 1, f"network2 had {len(network2.synapse_layers)} synapse layers, expected 1"
        assert len(network3.synapse_layers) == 0, f"network3 had {len(network2.synapse_layers)} synapse layers, expected 0"
        
        
    def test_network_builds_with_correct_dimensions(self):
        network = Network([2, 5, 1])

        assert len(network.node_layers[0]) == 2, f"network.layers[0] had {len(network.node_layers[0])} nodes, expected 2"
        assert len(network.node_layers[1]) == 5, f"network.layers[1] had {len(network.node_layers[1])} nodes, expected 5"
        assert len(network.node_layers[2]) == 1, f"network.layers[2] had {len(network.node_layers[2])} nodes, expected 1"
        
        assert len(network.synapse_layers) == 2
        
        
    def test_network_builds_with_synapse_connections(self):
        network = Network([1, 2])
        
        input_node_id = network.node_layers[0][0].id
        output_node_id_1 = network.node_layers[1][0].id
        output_node_id_2 = network.node_layers[1][1].id
        
        assert self.synapse_exists(network.synapse_layers[0], input_node_id, output_node_id_1)
        assert self.synapse_exists(network.synapse_layers[0], input_node_id, output_node_id_2)
    
    
    def test_network_builds_synapse_connections_for_hidden_layers(self):
        network = Network([2, 1, 2])
        
        hidden_layer_node_id = network.node_layers[1][0].id
        output_node_id_1 = network.node_layers[2][0].id
        output_node_id_2 = network.node_layers[2][1].id
        
        second_synapse_layer = network.synapse_layers[1]
        
        assert self.synapse_exists(second_synapse_layer, hidden_layer_node_id, output_node_id_1)
        assert self.synapse_exists(second_synapse_layer, hidden_layer_node_id, output_node_id_2)


    def test_network_clears_evaluation_state(self):
        network = Network([1, 1])
        
        network.node_layers[0][0].activation = 0.5
        network.node_layers[0][0].starting_input = 0.5
        
        network.node_layers[1][0].activation = 0.5
        network.node_layers[1][0].starting_input = 0.5
        
        network.clear_evaluation_state()
        
        assert network.node_layers[0][0].activation == None
        assert network.node_layers[0][0].starting_input == None
    
        assert network.node_layers[1][0].activation == None
        assert network.node_layers[1][0].starting_input == None
    
    
    def synapse_exists(self, synapse_layer, input_node_id, output_node_id):
        for synapse in synapse_layer:
            if synapse.input_node.id == input_node_id and synapse.output_node.id == output_node_id:
                return True

        return False

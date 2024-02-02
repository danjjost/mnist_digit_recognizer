from src.network import Network


# A more complicated neural network with known values used for testing. 
# The full diagram is available in test_network_evaluation.drawio
class ComplexNetworkCreator():

    def create_complex_network(self) -> Network:
        network = Network([2, 3, 2])
        
        self.set_node_layer_1(network)
        self.set_synapse_layer_1(network)
        
        self.set_node_layer_2(network)
        self.set_synapse_layer_2(network)
        
        self.set_node_layer_3(network)
        
        return network

    
    def set_node_layer_1(self, network: Network):
        network.node_layers[0][0].starting_input = 1.5
        network.node_layers[0][0].bias = 2

        network.node_layers[0][1].starting_input = 1.1
        network.node_layers[0][1].bias = 3
    
        
    def set_synapse_layer_1(self, network: Network):
        self.get_synapse_between(network, network.node_layers[0][0].id, network.node_layers[1][0].id).weight = 0.1
        self.get_synapse_between(network, network.node_layers[0][0].id, network.node_layers[1][1].id).weight = 0.4
        self.get_synapse_between(network, network.node_layers[0][0].id, network.node_layers[1][2].id).weight = 0.5
        
        self.get_synapse_between(network, network.node_layers[0][1].id, network.node_layers[1][0].id).weight = 0.2
        self.get_synapse_between(network, network.node_layers[0][1].id, network.node_layers[1][1].id).weight = 0.6
        self.get_synapse_between(network, network.node_layers[0][1].id, network.node_layers[1][2].id).weight = 1.2
    
    
    def set_node_layer_2(self, network: Network):
        network.node_layers[1][0].bias = 3
        network.node_layers[1][1].bias = 1
        network.node_layers[1][2].bias = 1.1
        
        
    def set_synapse_layer_2(self, network: Network):
        self.get_synapse_between(network, network.node_layers[1][0].id, network.node_layers[2][0].id).weight = 1.2
        self.get_synapse_between(network, network.node_layers[1][0].id, network.node_layers[2][1].id).weight = 1.4
        
        self.get_synapse_between(network, network.node_layers[1][1].id, network.node_layers[2][0].id).weight = 0.2
        self.get_synapse_between(network, network.node_layers[1][1].id, network.node_layers[2][1].id).weight = 0.4
        
        self.get_synapse_between(network, network.node_layers[1][2].id, network.node_layers[2][0].id).weight = 1.5
        self.get_synapse_between(network, network.node_layers[1][2].id, network.node_layers[2][1].id).weight = 1.3
        
        
    def set_node_layer_3(self, network: Network):
        network.node_layers[2][0].bias = 0.4
        network.node_layers[2][1].bias = 0.6
        
    
    def get_synapse_between(self, network: Network, input_node_id: str, output_node_id: str):
        for synapse_layer in network.synapse_layers:
            for synapse in synapse_layer:
                if synapse.input_node.id == input_node_id and synapse.output_node.id == output_node_id:
                    return synapse 
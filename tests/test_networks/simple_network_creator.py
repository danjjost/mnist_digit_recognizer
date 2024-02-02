from src.network import Network


# A simple 3 node network with known values used for testing.
# The full diagram is available in test_network_evaluation.drawio
class SimpleNetworkCreator():
    def create_simple_network(self):
        network = Network([2, 1])
        
        network.node_layers[0][0].starting_input = 1.5
        network.node_layers[0][0].bias = 2
        
        network.node_layers[0][1].starting_input = 1.1
        network.node_layers[0][1].bias = 3
        
        network.synapse_layers[0][0].weight = 0.3
        network.synapse_layers[0][1].weight = 0.4
        
        network.node_layers[1][0].bias = 1
        
        return network
        
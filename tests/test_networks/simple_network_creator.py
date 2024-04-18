from decimal import Decimal
from src.network import Network


# A simple 3 node network with known values used for testing.
# The full diagram is available in test_network_evaluation.drawio
class SimpleNetworkCreator():
    def create(self):
        network = Network([2, 1])
        
        network.node_layers[0][0].starting_input = Decimal('1.5')
        network.node_layers[0][0].bias = Decimal('2')
        
        network.node_layers[0][1].starting_input = Decimal('1.1')
        network.node_layers[0][1].bias = Decimal('3')
        
        network.synapse_layers[0][0].weight = Decimal('0.3')
        network.synapse_layers[0][1].weight = Decimal('0.4')
        
        network.node_layers[1][0].bias = Decimal('1')
        
        return network
        
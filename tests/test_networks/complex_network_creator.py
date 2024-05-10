from decimal import Decimal
from src.neuralnet.network import Network


# A more complicated neural network with known values used for testing. 
# The full diagram is available in test_network_evaluation.drawio
class ComplexNetworkCreator():

    def create(self) -> Network:
        network = Network([2, 3, 2])
        
        self.set_node_layer_1(network)
        self.set_synapse_layer_1(network)
        
        self.set_node_layer_2(network)
        self.set_synapse_layer_2(network)
        
        self.set_node_layer_3(network)
        
        return network

    
    def set_node_layer_1(self, network: Network):
        network.node_layers[0][0].starting_input = Decimal('1.5')
        network.node_layers[0][0].bias = Decimal('2')

        network.node_layers[0][1].starting_input = Decimal('1.1')
        network.node_layers[0][1].bias = Decimal('3')
    
        
    def set_synapse_layer_1(self, network: Network):
        network.get_synapse_between(network.node_layers[0][0].id, network.node_layers[1][0].id).weight = Decimal('0.1')
        network.get_synapse_between(network.node_layers[0][0].id, network.node_layers[1][1].id).weight = Decimal('0.4')
        network.get_synapse_between(network.node_layers[0][0].id, network.node_layers[1][2].id).weight = Decimal('0.5')
        
        network.get_synapse_between(network.node_layers[0][1].id, network.node_layers[1][0].id).weight = Decimal('0.2')
        network.get_synapse_between(network.node_layers[0][1].id, network.node_layers[1][1].id).weight = Decimal('0.6')
        network.get_synapse_between(network.node_layers[0][1].id, network.node_layers[1][2].id).weight = Decimal('1.2')
    
    
    def set_node_layer_2(self, network: Network):
        network.node_layers[1][0].bias = Decimal('3')
        network.node_layers[1][1].bias = Decimal('1')
        network.node_layers[1][2].bias = Decimal('1.1')
        
        
    def set_synapse_layer_2(self, network: Network):
        network.get_synapse_between(network.node_layers[1][0].id, network.node_layers[2][0].id).weight = Decimal('1.2')
        network.get_synapse_between(network.node_layers[1][0].id, network.node_layers[2][1].id).weight = Decimal('1.4')
        
        network.get_synapse_between(network.node_layers[1][1].id, network.node_layers[2][0].id).weight = Decimal('0.2')
        network.get_synapse_between(network.node_layers[1][1].id, network.node_layers[2][1].id).weight = Decimal('0.4')
        
        network.get_synapse_between(network.node_layers[1][2].id, network.node_layers[2][0].id).weight = Decimal('1.5')
        network.get_synapse_between(network.node_layers[1][2].id, network.node_layers[2][1].id).weight = Decimal('1.3')
        
        
    def set_node_layer_3(self, network: Network):
        network.node_layers[2][0].bias = Decimal('0.4')
        network.node_layers[2][1].bias = Decimal('0.6')
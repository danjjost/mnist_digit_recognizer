import json
from src.neuralnet.network import Network
from src.neuralnet.to_dict.network_to_dict import NetworkToDict
from azure.storage.blob import ContainerClient

from src.pipeline.population import PopulationDTO


class BlobClient:
    def __init__(self, network_to_dict: NetworkToDict, container: ContainerClient) -> None:
        self.network_to_dict = network_to_dict
        self.container = container

    def upload_population(self, population: PopulationDTO):
        self.upload_batch(population.population)

    def upload_batch (self, networks: list[Network]):
        for network in networks:
            self.upload_blob(network)        
    
    def upload_blob(self, network: Network):
        network_dictionary = self.network_to_dict.to_dict(network)
        network_json: str = json.dumps(network_dictionary)
        
        blob_client = self.container.get_blob_client(blob=network.id)
        blob_client.upload_blob(network_json, blob_type="BlockBlob", overwrite=True) # type: ignore

    def clear(self):
        blobs = self.container.list_blobs()
        for blob in blobs:
            self.container.delete_blob(blob.name)
            
    def delete(self, id: str):
        blob_client = self.container.get_blob_client(blob=id)
        blob_client.delete_blob()
        
    def download(self, id: str) -> Network:
        blob_client = self.container.get_blob_client(blob=id)
        network_json = blob_client.download_blob().content_as_text() #type: ignore
        network_dict = json.loads(network_json)
        return self.network_to_dict.from_dict(network_dict)
    
    def get_all_ids(self) -> list[str]:
        blobs = self.container.list_blobs()
        return [blob.name for blob in blobs]
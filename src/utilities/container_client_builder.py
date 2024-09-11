from azure_config import AzureConfig
from azure.storage.blob import BlobServiceClient, ContainerClient


class ContainerClientBuilder():
    def build(self, azure_config: AzureConfig, container_name: str) -> ContainerClient:
        connection_string = azure_config.neural_network_blob_connection_string
        
        if connection_string is None:
            raise ValueError("Neural network connection string was not found! Make sure to set it in your .env or environment variables.")
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        return blob_service_client.get_container_client(container_name)
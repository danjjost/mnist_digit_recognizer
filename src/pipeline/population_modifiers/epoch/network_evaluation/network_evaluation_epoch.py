

from src.azure.blob_client import BlobClient
from src.azure.blob_poller import BlobPoller
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.i_population_modifier import IPopulationModifier


class NetworkEvaluationEpoch(IPopulationModifier):
    def __init__(self, input_blob_client: BlobClient, output_blob_client: BlobClient, blob_poller: BlobPoller) -> None:
        self.input_blob_client = input_blob_client
        self.output_blob_client = output_blob_client
        self.blob_poller = blob_poller
            
    def run(self, population: PopulationDTO) -> PopulationDTO:
        self.input_blob_client.upload_batch(population.population)

        self.blob_poller.poll(population)

        self.input_blob_client.clear()
        self.output_blob_client.clear()

        return population
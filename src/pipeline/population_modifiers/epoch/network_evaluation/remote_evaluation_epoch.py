from src.azure_helpers.blob_client import BlobClient
from src.azure_helpers.blob_poller import BlobPoller
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.i_population_modifier import IPopulationModifier
from src.utilities.logger import Logger


class RemoteEvaluationEpoch(IPopulationModifier):
    def __init__(self, input_blob_client: BlobClient, output_blob_client: BlobClient, blob_poller: BlobPoller) -> None:
        self.input_blob_client = input_blob_client
        self.output_blob_client = output_blob_client
        self.blob_poller = blob_poller
            
    def run(self, population: PopulationDTO) -> PopulationDTO:
        Logger().debug(f'RemoteEvaluationEpoch - Starting...')
        
        Logger().debug(f'RemoteEvaluationEpoch - Uploading population to input blob...')
        self.input_blob_client.upload_batch(population.population)

        Logger().debug(f'RemoteEvaluationEpoch - Polling for updated blobs...')
        self.blob_poller.poll(population)

        Logger().debug(f'RemoteEvaluationEpoch - Clearing input and output blobs...')
        self.input_blob_client.clear()
        self.output_blob_client.clear()

        Logger().debug(f'RemoteEvaluationEpoch - Finished')
        return population
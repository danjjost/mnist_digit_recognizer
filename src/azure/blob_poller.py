from src.azure.blob_client import BlobClient
from src.pipeline.population import PopulationDTO


class BlobPoller():
    def __init__(self, output_blob_client: BlobClient) -> None:
        self.output_blob_client = output_blob_client

    def poll(self, population: PopulationDTO) -> None:
        raise NotImplementedError
    
        ## BLOB_POLLER
            # blobs_to_await = population.select(n=>n.id)
                    
            # while(blobs_to_await != empty && !timed_out)
                # wait for polling rate
                # get all blob names in output
                
                ## BLOB_UPDATER
                # foreach blob in output
                    # if blob in blobs_to_await
                        # download blob
                        # update blob in population
                
                # remove blob from blobs_to_await
from azure_config import AzureConfig
from config import Config
from src.azure_helpers.blob_client import BlobClient
from src.azure_helpers.blob_poller import BlobPoller
from src.digit_recognition.mnist_evaluation import MNISTEvaluation
from src.file.file_loader import FileLoader
from src.file.file_writer import FileWriter
from src.neuralnet.to_dict.network_to_dict import NetworkToDict
from src.pipeline.population_modifiers.cross.population_crosser import PopulationCrosser
from src.pipeline.population_modifiers.epoch.network_evaluation.remote_evaluation_epoch import RemoteEvaluationEpoch
from src.pipeline.population_modifiers.evolver.population_evolver import PopulationEvolver
from src.pipeline.pipeline import Pipeline
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.file.load_population import LoadPopulation
from src.pipeline.population_modifiers.file.save_population import SavePopulation
from src.pipeline.population_modifiers.population_mutator import PopulationMutator
from src.utilities.container_client_builder import ContainerClientBuilder

if __name__ == "__main__":
    # --- Config ---
    
    config = Config()
    azure_config = AzureConfig()


    # --- Dependencies ---
    
    network_to_dict = NetworkToDict()

    file_loader = FileLoader()
    file_writer = FileWriter()

    evaluation = MNISTEvaluation()
    
    input_container = ContainerClientBuilder().build(azure_config, "input")
    input_blob_client = BlobClient(network_to_dict, input_container)
    
    output_container = ContainerClientBuilder().build(azure_config, "output")
    output_blob_client = BlobClient(network_to_dict, output_container)
    
    blob_poller = BlobPoller(output_blob_client, config)


    # --- Population Modifiers ---

    load_population = LoadPopulation(config.input_file_path, network_to_dict, file_loader)

    epoch = RemoteEvaluationEpoch(input_blob_client, output_blob_client, blob_poller)

    mutator = PopulationMutator(config)
    evolver = PopulationEvolver(config.percent_predation)
    
    
    genetic_mutator = PopulationCrosser(config)

    save_population = SavePopulation(config.output_file_path, network_to_dict, file_writer)


    # end modifiers


    pipeline = Pipeline()

    pipeline.add(load_population)
    pipeline.add(epoch)
    pipeline.add(evolver)
    pipeline.add(genetic_mutator)
    pipeline.add(mutator)
    pipeline.add(save_population)

    while True:
        pipeline.run(PopulationDTO([]))
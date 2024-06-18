from config import Config
from src.digit_recognition.mnist_evaluation import MNISTEvaluation
from src.file.file_loader import FileLoader
from src.file.file_writer import FileWriter
from src.neuralnet.network_to_dict import NetworkToDict
from src.pipeline.evaluation_epoch import EvaluationEpoch
from src.pipeline.load_population import LoadPopulation
from src.pipeline.pipeline import Pipeline
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.population_evolver import PopulationEvolver
from src.pipeline.save_population import SavePopulation

if __name__ == "__main__":
    config = Config()
    network_to_dict = NetworkToDict()
    file_loader = FileLoader()
    file_writer = FileWriter()
    evaluation = MNISTEvaluation()


    # population modifiers

    load_population = LoadPopulation(config.input_file_path, network_to_dict, file_loader)

    evolver = PopulationEvolver()
    epoch = EvaluationEpoch(evaluation)

    save_population = SavePopulation(config.output_file_path, network_to_dict, file_writer)

    # end modifiers


    pipeline = Pipeline()

    pipeline.add(load_population)
    #pipeline.add(evolver)
    pipeline.add(epoch)
    pipeline.add(save_population)

    while True:
        pipeline.run(PopulationDTO([]))
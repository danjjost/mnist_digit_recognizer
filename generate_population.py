from config import Config
from src.pipeline.population_generator import PopulationGenerator
from src.pipeline.save_population import SavePopulation

if __name__ == "__main__":
    config = Config()    
    
    population = PopulationGenerator().generate(config.population_size, config.schema);
    print(population.population[0].node_layers[0][0].bias)
    SavePopulation(config.input_file_path).run(population)
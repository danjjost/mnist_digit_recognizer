from src.pipeline.population_generator import PopulationGenerator
from src.pipeline.save_population import SavePopulation

if __name__ == "__main__":
    population = PopulationGenerator().generate(100, [9, 18, 18, 9]);
    SavePopulation('./populations/population_input.json').run(population)
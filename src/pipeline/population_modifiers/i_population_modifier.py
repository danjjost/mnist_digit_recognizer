from src.pipeline.population import PopulationDTO

class IPopulationModifier:
    def run(self, population: PopulationDTO) -> PopulationDTO:
        raise NotImplementedError("PopulationModifier subclasses must implement run method")
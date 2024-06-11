from decimal import Decimal


class WeatherConditions():
    
    def __init__(self, temperature: Decimal, cloud_cover: Decimal):
        self.temperature = temperature
        self.cloud_cover = cloud_cover
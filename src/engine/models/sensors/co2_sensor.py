import logging
from datetime import datetime

from configurations.conf_generator import SensorType
from engine.models.core.location import Location
from engine.models.sensors.camera_data import CameraData
from engine.models.sensors.sensor import Sensor


class CO2Sensor(Sensor):
    _sensor_type: SensorType = SensorType.CO2

    def __init__(
        self,
        timestamp: datetime,
        location: Location,
        sensor_id: str,
    ):
        Sensor.__init__(self, timestamp, location, sensor_id)
        self._co2 = None
        if not self._co2:
            logging.warning(
                f"Sensor {self._sensor_id} of type {CO2Sensor.sensor_type} "
                f"is missing CO₂ concentration data!"
            )

    @property
    def unit(self):
        return {"co2" : "μg/m³"}
    
    @property
    def data(self):
        return {"co2" : round(self._co2, 2)}

    def next(self) -> None:
        pass

    def log(self) -> None:
        logging.debug(
            f'Sensor {self._sensor_id} of type {CO2Sensor.sensor_type} '
            f'reported CO₂ concentration: {self._co2:.2f} μg/m³.'
        )

    @property 
    def sensor_type(self):
        return self._sensor_type
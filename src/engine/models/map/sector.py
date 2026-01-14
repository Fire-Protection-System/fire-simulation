import random
from threading import Lock
import logging
from typing import List
import random
import json as jssonLib
from datetime import timedelta
import copy
import math

from src.generator import conf_generator
from src.engine.models.environment import fire_spread
from src.engine.models.map.fire_state import FireState
from src.engine.models.map.sector_state import SectorState
from src.engine.models.map.sector_type import SectorType

logger = logging.getLogger(__name__)

class Sector:
    def __init__(
        self,
        sector_id: int,
        row: int,
        column: int,
        sector_type: SectorType,
        initial_state: SectorState,
        fire_state=None,
        extinguish_level=0,
        fire_level=.0,
        burn_level=0,
        num_brigades=0,
        num_patrols=0,
        sensors=None,
        initial_temperature=None,
        adjacent_sectors=None,
        is_modified=False,
        coef_alpha=None
    ):
        self.lock = Lock()
        self._sector_id = sector_id
        self._row = row
        self._column = column
        self._sector_type = sector_type
        self._state = initial_state
        self._extinguish_level = extinguish_level
        self._fire_level = fire_level
        self._burn_level = burn_level
        self._number_of_fire_brigades = num_brigades
        self._number_of_forester_patrols = num_patrols
        self._sensors = sensors or []
        self._fire_state = fire_state or FireState.INACTIVE
        self._adjacent_sectors = adjacent_sectors  
        self._is_modified = is_modified  
        
        if initial_temperature is not None:
            self._initial_temperature = initial_temperature
        elif hasattr(initial_state, 'temperature'):
            self._initial_temperature = initial_state.temperature
            
        self._coef_alpha = coef_alpha or fire_spread.calculate_alpha(self._sector_type)

    @property
    def sector_id(self) -> int:
        return self._sector_id

    @property
    def row(self) -> int:
        return self._row

    @property
    def column(self) -> int:
        return self._column
    
    @property
    def sector_type(self) -> SectorType:
        return self._sector_type

    @property
    def state(self) -> SectorState:
        return self._state
    
    @property
    def fire_level(self) -> int:
        return self._fire_level

    @property
    def extinguish_level(self) -> int:
        return self._extinguish_level

    @property
    def burn_level(self) -> int:
        return self._burn_level
    
    @fire_level.setter
    def fire_level(self, fire):
        self._fire_level = fire
        self._is_modified = True

    @burn_level.setter
    def burn_level(self, burn):
        self._burn_level = burn
        self._is_modified = True

    @extinguish_level.setter
    def extinguish_level(self, extinguish):
        self._extinguish_level = extinguish
        self._is_modified = True

    @property
    def sensors(self):
        return self._sensors
    
    @property
    def fire_state(self) -> FireState:
        return self._fire_state
    
    @property
    def is_modified(self) -> bool:
        """Check if sector has been modified since last reset."""
        return self._is_modified
    
    def reset_modified_flag(self) -> None:
        """Reset the modified flag after state has been sent."""
        self._is_modified = False

    def add_sensor(self, sensor):
        self._sensors.append(sensor)
        self._is_modified = True

    def remove_sensor(self, sensor):
        self._sensors.remove(sensor)
        self._is_modified = True

    def update_fire(self, state: FireState, fireLevel: int):
        """Start fire with optimized random generation"""
        self._fire_level = fireLevel
        self._fire_state = state
        self._is_modified = True
    
    def start_fire(self):
        """Start fire in this sector"""
        if self._fire_state == FireState.INACTIVE:
            self._fire_level = random.randint(5, 20)
            self._fire_state = FireState.ACTIVE
            self._is_modified = True

    def update_sector(self):
        """Update sector state - fire level, burn level, extinguish level"""
        # Update extinguish level based on current fire brigades
        new_extinguish_level = self._number_of_fire_brigades * 5  # FIRE_FIGHTERS_MULTIPLIER
        if new_extinguish_level != self._extinguish_level:
            self._extinguish_level = new_extinguish_level
            self._is_modified = True

        if self._fire_state != FireState.ACTIVE:
            return
        
        # Update fire level
        fire_add = (self._fire_level / 10) * self._coef_alpha * 1  # FIRE_LEVEL_MULTIPLIER
        fire_sub = self._extinguish_level
        new_fire_level = min(self._fire_level + fire_add - fire_sub, 100)
        
        if new_fire_level <= 0:
            if self._fire_state != FireState.INACTIVE or self._fire_level != 0:
                self._fire_state = FireState.INACTIVE
                self._fire_level = 0
                self._is_modified = True
        else:
            if new_fire_level != self._fire_level:
                self._fire_level = new_fire_level
                self._is_modified = True
        
        # Update burn level
        fire_cubed = self._fire_level * self._fire_level * self._fire_level
        new_burn_level = min(self._burn_level + 0.00005 * fire_cubed, 100)
        
        if new_burn_level >= 100:
            if self._fire_state != FireState.LOST:
                self._fire_state = FireState.LOST
                self._fire_level = 0
                self._extinguish_level = 0
                self._is_modified = True
                logger.warning(f"Sector {self._sector_id} is lost!")
        else:
            if new_burn_level != self._burn_level:
                self._burn_level = new_burn_level
                self._is_modified = True

    def update_sensors(self):
        """Update all sensors with current sector state"""
        for sensor in self.sensors:
            sensor._timestamp += timedelta(seconds=1)
            if hasattr(sensor, '_pm2_5'):
                sensor._pm2_5 = self._state.pm2_5_concentration + random.uniform(-0.5, 0.5)
            if hasattr(sensor, '_temperature'):
                sensor._temperature = self._state.temperature + random.uniform(-0.5, 0.5)
                sensor._humidity = self._state.air_humidity + random.uniform(-0.5, 0.5)
            if hasattr(sensor, '_litter_moisture'):
                sensor._litter_moisture = self._state.plant_litter_moisture + random.uniform(-0.5, 0.5)
            if hasattr(sensor, '_co2'):
                sensor._co2 = self._state.co2_concentration + random.uniform(-1.0, 1.0)
            if hasattr(sensor, '_wind_speed'):
                sensor._wind_speed = self._state.wind_speed + random.uniform(-0.3, 0.3)
            if hasattr(sensor, '_wind_direction'):
                sensor._wind_direction = self._state.wind_direction

            try:
                from src.engine.models.sensors.camera import Camera
                if isinstance(sensor, Camera):
                    sensor._camera_data.smoke_detected = 1 if self._fire_level > 0 else 0
                    sensor._camera_data.smoke_level = float(self._fire_level)
            except Exception:
                pass
            
            # Removed sensor.log() call - logging is handled by telemetry system
            # This reduces unnecessary method calls on every tick
        jsons_by_type = {}
        for sensor in self.sensors:
            json = {
                "sensorId": sensor.sensor_id,
                "timestamp": sensor._timestamp.strftime('%Y-%m-%dT%H:%M:%S'),
                "sensorType": sensor.sensor_type.name,
                "location": {
                    "longitude": sensor._location.longitude,
                    "latitude": sensor._location.latitude,
                },
                "data": sensor.data
            }
            
            if sensor.sensor_type.name not in jsons_by_type:
                jsons_by_type[sensor.sensor_type.name] = []
            
            jsons_by_type[sensor.sensor_type.name].append(json)
        
        return jsons_by_type

    def make_sector_json(self):
        sector_json = {
            "sectorId":         int(self.sector_id), 
            # "fireState":      0,
            "fireLevel":        float(self.fire_level),
            "burnLevel":        float(self.burn_level), 
            "extinguishLevel":  float(self.extinguish_level)
        }
        # Only log significant fire events (changed to DEBUG to reduce log volume)
        return sector_json

    def clone(self):
        cloned = Sector(
            sector_id=self._sector_id,
            row=self._row,
            column=self._column,
            sector_type=self._sector_type,
            initial_state=copy.deepcopy(self._state),
            fire_state=self._fire_state,
            extinguish_level=self._extinguish_level,
            fire_level=self._fire_level,
            burn_level=self._burn_level,
            num_brigades=self._number_of_fire_brigades,
            num_patrols=self._number_of_forester_patrols,
            sensors=copy.deepcopy(self._sensors),
            initial_temperature=self._initial_temperature,
            is_modified=self._is_modified,
            coef_alpha=self._coef_alpha
        )
        
        return cloned
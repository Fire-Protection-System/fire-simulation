import random
from threading import Lock
import logging
from typing import Any, List, Optional
import random
import json as jssonLib
from datetime import timedelta
import copy
import math

from src.engine.models.environment import fire_spread
from src.engine.models.map.fire_state import FireState
from src.engine.models.map.sector_state import SectorState
from src.engine.models.map.sector_type import SectorType
from src.engine.models.sensors.sensor_type import SensorType

logger = logging.getLogger(__name__)

'''
    MAIN CLASS FOR SECTORS
    Sector class represents a single sector on the map.
    It looks awful, works awful, but it works. Sorry for that.
'''

class Sector:
    def __init__(
        self,
        sector_id: int,
        row: int,
        column: int,
        sector_type: SectorType,
        initial_state: SectorState,
        fire_state: Optional[FireState]            = None,
        extinguish_level: float                    = 0,
        fire_level: float                          = 0.0,
        burn_level: float                          = 0,
        num_brigades: int                          = 0,
        num_patrols: int                           = 0,
        sensors: Optional[List[Any]]               = None,
        initial_temperature: Optional[float]       = None,
        adjacent_sectors: Optional[List["Sector"]] = None,
        is_modified: bool                          = False,
        coef_alpha: Optional[float]                = None,
    ) -> None:
        self.lock = Lock()
        self._sector_id: int                   = sector_id
        self._row: int                         = row
        self._column: int                      = column
        self._sector_type: SectorType          = sector_type
        self._state: SectorState               = initial_state
        self._extinguish_level: float          = extinguish_level
        self._fire_level: float                = fire_level
        self._burn_level: float                = burn_level
        self._number_of_fire_brigades: int     = num_brigades
        self._number_of_forester_patrols: int  = num_patrols
        self._sensors: List[Any]               = sensors or []
        self._fire_state: FireState            = fire_state or FireState.INACTIVE
        self._adjacent_sectors: List["Sector"] = adjacent_sectors
        self._is_modified: bool                = is_modified

        self._initial_temperature = initial_temperature if initial_temperature is not None else 20.0
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
    def fire_level(self) -> float:
        return self._fire_level

    @property
    def extinguish_level(self) -> int:
        return self._extinguish_level

    @property
    def burn_level(self) -> int:
        return self._burn_level
    
    @fire_level.setter
    def fire_level(self, fire):
        """
        Set fire level, clamped to a non-negative value.
        Negative values can appear due to numerical effects of repeated
        extinguishing – from the simulation point of view anything <= 0
        means "no fire".
        """
        try:
            fire_val = float(fire)
        except (TypeError, ValueError):
            fire_val = 0.0

        # Clamp to [0, +inf)
        if fire_val < 0.0:
            fire_val = 0.0

        self._fire_level = fire_val
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
        if self._fire_state != FireState.ACTIVE:
            return
        
        ''' Update fire level (agent-driven extinguishing; only growth here).
            Slowed down so fire does not instantly take over the whole map.
        ''' 
        # Previously: (self._fire_level / 10) * alpha
        fire_add = (self._fire_level / 20.0) * self._coef_alpha
        new_fire_level = min(self._fire_level + fire_add, 100)
        
        if new_fire_level != self._fire_level:
            self._fire_level = new_fire_level
            self._is_modified = True
        
        fire_cubed = self._fire_level * self._fire_level * self._fire_level
        # Burn progression slowed down ~2.5x to reduce sectors being "lost" too quickly
        new_burn_level = min(self._burn_level + 0.00002 * fire_cubed, 100)
        
        if new_burn_level >= 100:
            self._fire_state = FireState.LOST
            self._fire_level = 0
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

            match sensor.sensor_type:
                case SensorType.PM2_5:
                    sensor._pm2_5 = self._state.pm2_5_concentration + random.uniform(-0.5, 0.5)
                
                case SensorType.TEMPERATURE_AND_AIR_HUMIDITY:
                    sensor._temperature = self._state.temperature + random.uniform(-0.5, 0.5)
                    sensor._humidity = self._state.air_humidity + random.uniform(-0.5, 0.5)
                
                case SensorType.LITTER_MOISTURE:
                    sensor._litter_moisture = self._state.plant_litter_moisture + random.uniform(-0.5, 0.5)
               
                case SensorType.CO2:
                    sensor._co2 = self._state.co2_concentration + random.uniform(-1.0, 1.0)

                case SensorType.WIND_SPEED:
                    sensor._wind_speed = self._state.wind_speed + random.uniform(-0.3, 0.3)

                case SensorType.WIND_DIRECTION:
                    sensor._wind_direction = self._state.wind_direction

                case SensorType.CAMERA:
                    if hasattr(sensor, "_camera_data"):
                        sensor._camera_data.smoke_detected = 1 if self._fire_level > 0 else 0
                        sensor._camera_data.smoke_level = float(self._fire_level)
                case _:
                    pass
            
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
        return sector_json

    '''
    Clone sector - copying sector state to another sector.
    DEPRECIATED: Was used for copying sector state to another sector for MCTS calculations.

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
    '''
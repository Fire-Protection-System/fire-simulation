import json
import random

from datetime import datetime
from typing import TypeAlias
from typing import Tuple
from typing import List, Tuple, Optional

from simulation.sectors.sector import Sector
from simulation.location import Location
from simulation.sectors.sector_state import SectorState
from simulation.sectors.sector_type import SectorType
from simulation.sectors.geographic_direction import GeographicDirection
from simulation.sensors.temperature_and_air_humidity_sensor import TemperatureAndAirHumiditySensor
from simulation.sensors.wind_speed_sensor import WindSpeedSensor
from simulation.sensors.wind_direction_sensor import WindDirectionSensor
from simulation.sensors.co2_sensor import CO2Sensor
from simulation.sensors.litter_moisture_sensor import LitterMoistureSensor
from simulation.sensors.pm2_5_sensor import PM2_5Sensor
from simulation.cameras.camera import Camera
from simulation.forester_patrols.forester_patrol import ForesterPatrol
from simulation.fire_brigades.fire_brigade import FireBrigade
from simulation.fire_brigades.fire_brigade_state import FIREBRIGADE_STATE
from simulation.forester_patrols.forest_patrols_state import FORESTERPATROL_STATE
from simulation.sectors.fire_state import FireState

ForestMapCornerLocations: TypeAlias = tuple[Location, Location, Location, Location] 


class ForestMap:
    def __init__(
        self,
        forest_id: str,
        forest_name: str,
        rows: int,
        columns: int,
        location: ForestMapCornerLocations,
        sectors: list[list[Sector]],
        foresterPatrols: list[ForesterPatrol],
        fireBrigades: list[FireBrigade]
    ):
        self._forest_id = forest_id
        self._forest_name = forest_name
        self._rows = rows
        self._columns = columns
        self._location = location
        self._sectors = sectors
        self._forester_patrols = foresterPatrols
        self._fire_brigades = fireBrigades

    @classmethod
    def from_conf(cls, conf):
        # Przetwórz dane lokalizacji
        location = cls._parse_locations(conf["location"])

        # Przetwórz sektory
        sectors = cls._parse_sectors(conf)

        # Oblicz parametry mapy na podstawie lokalizacji
        bounds = cls._calculate_bounds(location, conf["rows"], conf["columns"])

        # Dodaj sensory do odpowiednich sektorów
        cls._assign_sensors_to_sectors(conf["sensors"], sectors, bounds)

        cls._assign_cameras_to_sectors(conf["cameras"], sectors, bounds)

        brigades = cls._parse_fire_brigades(conf)
        patrols = cls._parse_forester_patrols(conf)
        

        # Stwórz i zwróć obiekt ForestMap
        return cls(
            forest_id=conf["forestId"],
            forest_name=conf["forestName"],
            rows=conf["rows"],
            columns=conf["columns"],
            location=location,
            sectors=sectors,
            foresterPatrols=patrols,
            fireBrigades=brigades
        )

    @staticmethod
    def _parse_locations(locations_conf):
        return tuple(Location(**location) for location in locations_conf)

    @staticmethod
    def _parse_sectors(conf):        
        json_conf = json.dumps(conf, indent=4)

        sectors = [[None for _ in range(conf["columns"])] for _ in range(conf["rows"])]
        for val in conf["sectors"]:
            initial_state = SectorState(
                temperature=val["initialState"]["temperature"],
                wind_speed=val["initialState"]["windSpeed"],
                wind_direction=GeographicDirection[val["initialState"]["windDirection"]],
                air_humidity=val["initialState"]["airHumidity"],
                plant_litter_moisture=val["initialState"]["plantLitterMoisture"],
                co2_concentration=val["initialState"]["co2Concentration"],
                pm2_5_concentration=val["initialState"]["pm2_5Concentration"],
            )
            sectors[val["row"] - 1][val["column"] - 1] = Sector(
                sector_id=val["sectorId"],
                row=val["row"] - 1,
                column=val["column"] - 1,
                sector_type=SectorType[val["sectorType"]],
                initial_state=initial_state,
                fire_level=val["initialState"]["fireLevel"],
                fire_state= FireState.ACTIVE if (val["initialState"]["fireLevel"] > 0) else FireState.INACTIVE
            )
        return sectors
    
    def _parse_fire_brigades(conf):
        fire_brigades = []
        for fb_data in conf["fireBrigades"]:
            fire_brigade_id = fb_data["fireBrigadeId"]
            timestamp = datetime.fromisoformat(fb_data["timestamp"]) 
            state = FIREBRIGADE_STATE[fb_data["state"]]
            base_location = Location(
                longitude=float(fb_data["baseLocation"]["longitude"]),
                latitude=float(fb_data["baseLocation"]["latitude"])
            )
            current_location = Location(
                longitude=float(fb_data["currentLocation"]["longitude"]),
                latitude=float(fb_data["currentLocation"]["latitude"])
            )

            fire_brigades.append(FireBrigade(
                fire_brigade_id=fire_brigade_id,
                timestamp=timestamp,
                initial_state=state,
                base_location=base_location,
                initial_location=current_location
            ))

        return fire_brigades
    
    def _parse_forester_patrols(conf):
        foresterPatrols = []
        for fb_data in conf["foresterPatrols"]:
            forester_patrol_id = fb_data["foresterPatrolId"]
            timestamp = datetime.fromisoformat(fb_data["timestamp"]) 
            state = FORESTERPATROL_STATE[fb_data["state"]]
            base_location = Location(
                longitude=float(fb_data["baseLocation"]["longitude"]),
                latitude=float(fb_data["baseLocation"]["latitude"])
            )
            current_location = Location(
                longitude=float(fb_data["currentLocation"]["longitude"]),
                latitude=float(fb_data["currentLocation"]["latitude"])
            )

            foresterPatrols.append(ForesterPatrol(
                forester_patrol_id=forester_patrol_id,
                timestamp=timestamp,
                initial_state=state,
                base_location=base_location,
                initial_location=current_location
            ))

        return foresterPatrols

    @staticmethod
    def _calculate_bounds(locations, rows, columns):
        min_lat = min(location.latitude for location in locations)
        min_lon = min(location.longitude for location in locations)
        diff_lat = max(location.latitude for location in locations) - min_lat
        diff_lon = max(location.longitude for location in locations) - min_lon
        return {
            "min_lat": min_lat,
            "min_lon": min_lon,
            "width_sectors": diff_lon / columns,
            "height_sectors": diff_lat / rows
        }

    
    @staticmethod
    def _assign_sensors_to_sectors(sensors, sectors, bounds):
        for sensor in sensors:
            sensor_obj = ForestMap._create_sensor(sensor)
            if not sensor_obj:
                continue

            sensor_location = Location(**sensor["location"])
            row = int((sensor_location.latitude - bounds["min_lat"]) / bounds["height_sectors"])
            column = int((sensor_location.longitude - bounds["min_lon"]) / bounds["width_sectors"])

            if 0 <= row < len(sectors) and 0 <= column < len(sectors[0]) and sectors[row][column]:
                sectors[row][column].add_sensor(sensor_obj)

    def _assign_cameras_to_sectors(cameras, sectors, bounds):
        for camera in cameras:
            camera_obj = ForestMap._create_camera(camera)

            if not camera_obj:
                continue

            camera_location = Location(**camera["location"])
            row = int((camera_location.latitude - bounds["min_lat"]) / bounds["height_sectors"])
            column = int((camera_location.longitude - bounds["min_lon"]) / bounds["width_sectors"])

            if 0 <= row < len(sectors) and 0 <= column < len(sectors[0]) and sectors[row][column]:
                sectors[row][column].add_sensor(camera_obj)

    @staticmethod
    def _create_sensor(sensor_conf):
        sensor_arguments = {
            "timestamp": datetime.now(),
            "location": Location(sensor_conf["location"]["latitude"], sensor_conf["location"]["longitude"]),
            "sensor_id": sensor_conf["sensorId"],
        }
        match sensor_conf["sensorType"]:
            case "TEMPERATURE_AND_AIR_HUMIDITY":
                return TemperatureAndAirHumiditySensor(**sensor_arguments)
            case "WIND_SPEED":
                return WindSpeedSensor(**sensor_arguments)
            case "WIND_DIRECTION":
                return WindDirectionSensor(**sensor_arguments)
            case "LITTER_MOISTURE":
                return LitterMoistureSensor(**sensor_arguments)
            case "PM2_5":
                return PM2_5Sensor(**sensor_arguments)
            case "CO2":
                return CO2Sensor(**sensor_arguments)
            case _:
                return None

    @staticmethod
    def _create_camera(camera_conf):
        return Camera(datetime.now(), Location(camera_conf["location"]["latitude"], camera_conf["location"]["longitude"]), camera_conf["cameraId"])


    @property
    def foresterPatrols(self):
        return self._forester_patrols
    
    @property
    def fireBrigades(self):
        return self._fire_brigades

    @property
    def forest_id(self) -> str:
        return self._forest_id

    @property
    def forest_name(self) -> str:
        return self._forest_name

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def columns(self) -> int:
        return self._columns

    @property
    def location(self) -> ForestMapCornerLocations:
        return self._location

    @property
    def sectors(self) -> list[list[Sector]]:
        return self._sectors
    
    def start_new_fire(self) -> Sector:
        row = random.choice(self.sectors)
        sector = random.choice(row)
        sector.start_fire()

        return sector        
    
    def get_sector_with_max_burn_level(self) -> Sector:
        max_burn_level = 0
        max_burn_sector = None
        for row in self._sectors:
            for sector in row:
                if sector.burn_level > max_burn_level:
                    max_burn_level = sector.burn_level
                    max_burn_sector = sector
        return max_burn_sector
    
    def get_sector_location(self, sector: Sector) -> Location:
        """
        Compute the geographic center of a given sector based on the four map corners.
        Assumes self._location is (lower-left, lower-right, upper-right, upper-left).
        """

        ul = self._location[3]
        ur = self._location[2]
        lr = self._location[1]
        ll = self._location[0]

        total_lon_span = ur.longitude - ul.longitude
        total_lat_span = ul.latitude - ll.latitude  

        sector_width = total_lon_span / self._columns
        sector_height = total_lat_span / self._rows

        center_lon = ul.longitude + (sector.column + 0.5) * sector_width
        center_lat = ul.latitude - (sector.row + 0.5) * sector_height

        return Location(longitude=center_lon, latitude=center_lat)

    def get_sector(self, sector_id: int) -> Sector:
        for row in self._sectors:
            for sector in row:
                if sector.sector_id == sector_id:
                    return sector
        return None

    def find_sector(self, location: Location):
        bottom_left = self._location[0]
        bottom_right = self._location[1]
        top_right = self._location[2]
        top_left = self._location[3]

        min_lat = bottom_left.latitude
        max_lat = top_left.latitude
        min_lon = bottom_left.longitude
        max_lon = bottom_right.longitude

        lat_diff = max_lat - min_lat
        lon_diff = max_lon - min_lon

        if lat_diff == 0 or lon_diff == 0:
            return None  

        lat_interpolation = (location.latitude - min_lat) / lat_diff
        lon_interpolation = (location.longitude - min_lon) / lon_diff

        height_index = int((1 - lat_interpolation) * self.rows)  # flip latitude (north = lower index)
        width_index = int(lon_interpolation * self.columns)

        height_index = max(0, min(self.rows - 1, height_index))
        width_index = max(0, min(self.columns - 1, width_index))

        return self._sectors[height_index][width_index]


    def get_adjacent_sectors(self, sector: Sector) -> list[Tuple[Sector, GeographicDirection]]:
        row = sector.row
        column = sector.column
        adjacent_sectors = []

        directions = [
            (-1, 0, GeographicDirection.N),
            (-1, 1, GeographicDirection.NE),
            (0, 1, GeographicDirection.E),
            (1, 1, GeographicDirection.SE),
            (1, 0, GeographicDirection.S),
            (1, -1, GeographicDirection.SW),
            (0, -1, GeographicDirection.W),
            (-1, -1, GeographicDirection.NW)
        ]

        for delta_row, delta_column, direction in directions:
            new_row = row + delta_row
            new_column = column + delta_column

            if 0 <= new_row < len(self.sectors) and 0 <= new_column < len(self.sectors[new_row]):
                adjacent_sectors.append((self.sectors[new_row][new_column], direction))

        return adjacent_sectors

    def update_sectors(self, new_sectors: List[Sector]):
        id_map = {s.sector_id: s for s in new_sectors}
        for row in self.sectors:
            for i, s in enumerate(row):
                row[i] = id_map[s.sector_id]

    def clone(self) -> 'ForestMap':
        cloned_sectors = [
            [sector.clone() for sector in row]
            for row in self._sectors
        ]

        cloned_brigades = [brigade.clone() for brigade in self._fire_brigades]
        cloned_patrols = [patrol.clone() for patrol in self._forester_patrols]

        return ForestMap(
            forest_id=self._forest_id,
            forest_name=self._forest_name,
            rows=self._rows,
            columns=self._columns,
            location=tuple(Location(loc.latitude, loc.longitude) for loc in self._location),
            sectors=cloned_sectors,
            foresterPatrols=cloned_patrols,
            fireBrigades=cloned_brigades
        )
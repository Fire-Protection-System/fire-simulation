import logging
from datetime import datetime
import json

from simulation.sectors.sector import Sector
from simulation.agent import Agent
from simulation.location import Location
from simulation.agent_state import AGENT_STATE

class FireBrigade(Agent):
    def __init__(
        self,
        fire_brigade_id: str,
        timestamp: datetime,
        initial_state: AGENT_STATE,
        base_location: Location,
        initial_location: Location,
    ):
        Agent.__init__(self, timestamp, base_location, initial_location)
        self._fire_brigade_id = fire_brigade_id
        self._state = initial_state
        self._destination = initial_location

        self._initial_location = initial_location
        self._base_location = base_location
        self._timestamp = timestamp

    @property
    def fire_brigade_id(self) -> str:
        return self._fire_brigade_id
    
    def is_task_finished(self, sector: Sector) -> bool:
        return sector.fire_level <= 0
    
    def increment_agents_in_sector(self, sector):
        sector._number_of_fire_brigades += 1

    def decrement_agents_in_sector(self, sector):
        sector._number_of_fire_brigades -= 1

    @property
    def getId(self):
        return self.fire_brigade_id
    
    @property
    def initial_state(self) -> AGENT_STATE:
        return self._state

    def next(self):
        pass

    def log(self) -> None:
        print(f'Fire brigade {self._fire_brigade_id} is in state: {self._state}.')
        logging.debug(f'Fire brigade {self._fire_brigade_id} is in state: {self._state}.')

    def clone(self) -> 'FireBrigade':
        return FireBrigade(
            fire_brigade_id=self._fire_brigade_id,
            timestamp=self._timestamp, 
            initial_state=self._state,
            base_location=Location(self._base_location.latitude, self._base_location.longitude),
            initial_location=Location(self._initial_location.latitude, self._initial_location.longitude)  # Assuming _initial_location exists
        )

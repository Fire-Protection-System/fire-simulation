import random
import logging
from typing import Dict, Any, List, Optional

from src.engine.base import SimulationEngine
from src.engine.models.map.forest_map import ForestMap
from src.engine.models.map.sector import Sector
from src.engine.models.map.fire_state import FireState
from src.engine.models.environment.wind import Wind
from src.engine.models.environment.fire_spread import calculate_beta
from src.engine.agent_manager.agent_manager import AgentManager

logger = logging.getLogger(__name__)

class SimpleSimulationEngine(SimulationEngine):
    def __init__(self):
        self.config: Optional[dict] = None
        self._map: Optional[ForestMap] = None
        self.agents_manager: Optional[AgentManager] = None
        self.wind: Optional[Wind] = None
        self.all_sectors: List[Sector] = []
        self.sectors_on_fire: List[Sector] = []
        self._running = False
        self._tick_count = 0

    async def load_config(self, configuration: dict) -> None:
        logger.info("Loading configuration into SimpleSimulationEngine")
        self.config = configuration
        self._map = ForestMap.from_conf(configuration)
        self.agents_manager = AgentManager(self._map, configuration.get('agents', [])) 
        self.all_sectors = [s for row in self._map.sectors for s in row]
        self.wind = Wind()

        first = self._map.start_new_fire()
        
        if first:
            self.sectors_on_fire = [first]
            logger.info(f"Initial fire started at sector: {first}")
        else:
            self.sectors_on_fire = []
            logger.warning("No initial fire started")
        
        self._tick_count = 0

    async def start(self) -> None:
        if not self._map:
            raise ValueError("Configuration not loaded. Call load_config first.")
        self._running = True
        logger.info("Simulation started")

    async def stop(self) -> None:
        self._running = False
        logger.info("Simulation stopped")

    async def step(self, ticks: int = 1) -> Dict[str, Any]:
        """
        Update Simulation step. Return dict:
        {
            "tick": current_tick,
            "sensor_messages": { sensor_type_name: [json_obj, ...], ... },
            "sector_states": [json_obj, ...],
            "agent_states": [json_obj, ...],
            "events": [...]
        }
        """

        out_messages: Dict[str, List[Any]] = {}
        sector_states: List[Any] = []
        events: List[Any] = []
        agent_states: List[Any] = []

        if not (self._map and self.wind and self.agents_manager):
            logger.warning("Simulation not fully initialized, returning empty step result")
            return {
                "tick": self._tick_count,
                "sensor_messages": out_messages, 
                "sector_states": sector_states,
                "agent_states": agent_states,
                "events": events
            }

        for _ in range(ticks):
            self._tick_count += 1
            
            new_sectors_on_fire: List[Sector] = []
            for sector in list(self.sectors_on_fire):
                sector.update_sector()
                neighbours = self._map.get_adjacent_sectors(sector)  
                for neighbour, direction in neighbours:
                    if neighbour.fire_state is FireState.INACTIVE:
                        prob = calculate_beta(self.wind, neighbour.sector_type, direction)
                        if random.random() < prob:
                            neighbour.start_fire()
                            new_sectors_on_fire.append(neighbour)
                            logger.debug(f"Fire spread to sector at tick {self._tick_count}")

            self.sectors_on_fire = [s for s in self.sectors_on_fire if s.fire_state is FireState.ACTIVE]
            self.sectors_on_fire.extend(new_sectors_on_fire)
            self.wind.update_wind()
            self.agents_manager.update_agents_states()

            for sector in self.all_sectors:
                sector.update_sensors()
                for sensor_type, jsons in sector.make_jsons().items():
                    out_messages.setdefault(sensor_type, []).extend(jsons)
                sector_states.append(sector.make_sector_json())

        logger.info(f"Step completed. Tick: {self._tick_count}, Active fires: {len(self.sectors_on_fire)}")
        
        return {
            "tick": self._tick_count,
            "sensor_messages": out_messages, 
            "sector_states": sector_states,
            "agent_states": agent_states,
            "events": events
        }

    async def pause(self) -> None:
        raise NotImplementedError("Pause functionality is not implemented yet.")

    def snapshot(self) -> Dict[str, Any]:
        return {
            "tick": self._tick_count,
            "running": self._running,
            "fire_count": len(self.sectors_on_fire),
            "total_sectors": len(self.all_sectors),
            "config_loaded": self.config is not None
        }

    def is_running(self) -> bool:
        return self._running
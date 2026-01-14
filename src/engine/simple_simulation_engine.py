import random
import logging
from typing import Dict, Any, List, Optional

from src.settings.communucation_settings import CommunicationSettings
from src.settings.simulation_settings import SimulationSettings

from src.engine.base import SimulationEngine
from src.engine.models.map.forest_map import ForestMap
from src.engine.models.map.sector import Sector
from src.engine.models.map.fire_state import FireState
from src.engine.models.environment.wind import Wind
from src.engine.models.environment.fire_spread import calculate_beta
from src.engine.agent_manager.agent_manager import AgentManager

logger = logging.getLogger(__name__)

class SimpleSimulationEngine(SimulationEngine):
    '''
    A simple simulation engine for fire simulation.
    Idea was to create standarized engine template that can be later extended with more complex features.

    Extra features could include:
        * Wind models [see. Wind interface]
        * Fire spread models [see. calculate_beta function]
        * Advanced agent management strategies

    Simple engine provides basic functionality, previously implemented in the project. 
    This should allow easy extension and allow plugable components/models, or swapping the whole engine.
    '''
    def __init__(
        self, 
        simulation_settings: SimulationSettings       = SimulationSettings(), 
        communication_settings: CommunicationSettings = CommunicationSettings()
    ):
        logger.info("Initializing SimpleSimulationEngine")
        logger.info(f"Simulation settings: {simulation_settings}")
        logger.info(f"Communication settings: {communication_settings}")
        
        self.simulation_settings: SimulationSettings        = simulation_settings
        self.communication_settings: CommunicationSettings  = communication_settings
        self.config: Optional[dict]                         = None
        self.agents_manager: Optional[AgentManager]         = None
        self.wind: Optional[Wind]                           = None
        self.all_sectors: List[Sector]                      = []
        self.sectors_on_fire: List[Sector]                  = []

        # Internal state
        self._tick_count: float                             = simulation_settings.tick_interval
        self._running: bool                                 = False 
        self._map: Optional[ForestMap]                      = None

    async def load_config(self, configuration: dict) -> None:
        logger.info("Loading configuration into SimpleSimulationEngine")

        try:
            self.config = configuration
            self._map = ForestMap.from_conf(configuration)
            self._tick_count = 0 # Todo check if should be loaded from config

            self.agents_manager = AgentManager(
                forest_map    = self._map, 
                message_store = None,
                engine        = self)
            
            self.speed_factor = 1.0
            self.all_sectors = [s for row in self._map.sectors for s in row]
            self.wind = Wind()
            await self.start_new_fire(0, 0)

            logger.info("="*50)
            logger.info("= Configuration Loaded Successfully =")
            logger.info("="*50)

            logger.info("ForestMap created: %dx%d sectors", self._map._rows, self._map._columns)
            logger.info("AgentManager initialized with %d brigades and %d patrols", len(self._map._fire_brigades), len(self._map._forester_patrols))
            logger.info("Initial fire started at sector (0,0)")
            logger.info("Configuration loading complete")

            # Debug info    
            logger.debug("="*50)
            logger.debug("= Configuration loading debug info =")
            logger.debug("="*50)

            logger.debug("Total sectors loaded: %d", len(self.all_sectors))
            logger.debug("Wind system initialized")
            logger.debug("Forest=%s, Rows=%s, Cols=%s", configuration.get("forestName", "unknown"),  configuration.get("rows"), configuration.get("columns"))

        except Exception as e:
            logger.error("Failed to load configuration: %s", e)
            raise e


    async def start(self) -> None:
        if not self._map:
            raise ValueError("Configuration not loaded. Call load_config first.")
        
        self._running = True
        # CommandConsumer will be started by EngineRunner after queues are set up
        logger.info("Simulation started")

    async def stop(self) -> None:
        """
        Stop simulation and reset internal state so a new run starts cleanly.
        """
        self._running = False
        self.sectors_on_fire = []
        self._tick_count = 0
        logger.info("Simulation stopped and internal state reset (tick_count=0, sectors_on_fire cleared)")

    def set_speed_factor(self, factor: float) -> None:
        if factor <= 0:
            raise ValueError("speed_factor must be > 0")
        logger.info("Setting simulation speed_factor to %s", factor)
        self.speed_factor = float(factor)
        
    async def start_new_fire(self, row: int, column: int) -> None:
        if not self._map:
            raise ValueError("Configuration not loaded. Call load_config first.")
        sector = self._map.sectors[row][column]
        if sector.fire_state is FireState.INACTIVE:
            sector.update_fire(FireState.ACTIVE, random.randint(5, 20))
            self.sectors_on_fire.append(sector)
            logger.info(f"New fire started at sector: row={row}, col={column}, id={sector.sector_id}")
        else:
            logger.warning(f"Sector at row={row}, col={column} is already on fire or active.")

    def step(self, ticks: int = 1) -> Dict[str, Any]:
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
            logger.warning(f"Map: {self._map is not None}, Wind: {self.wind is not None}, AgentManager: {self.agents_manager is not None}")
            
            return {
                "tick": self._tick_count,
                "sensor_messages": out_messages, 
                "sector_states": sector_states,
                "agent_states": agent_states,
                "events": events
            }

        for tick_idx in range(ticks):
            self._tick_count += 1
            
            new_sectors_on_fire: List[Sector] = []
            
            # Optimized: Only update sectors that are active or have brigades
            # This significantly reduces computation for large sector counts
            sectors_to_update = []
            active_sectors_for_spread = []
            
            for sector in self.all_sectors:
                # Only update sectors that are on fire, have brigades, or are adjacent to fires
                if (sector.fire_state == FireState.ACTIVE or 
                    sector._number_of_fire_brigades > 0 or
                    sector._number_of_forester_patrols > 0):
                    sectors_to_update.append(sector)
                    if sector.fire_state == FireState.ACTIVE:
                        active_sectors_for_spread.append(sector)
            
            # Batch update active sectors
            for sector in sectors_to_update:
                sector.update_sector()
            
            # Fire spread: only check neighbors of active sectors
            for sector in active_sectors_for_spread:
                neighbours = self._map.get_adjacent_sectors(sector)  
                for neighbour, direction in neighbours:
                    if neighbour.fire_state is FireState.INACTIVE:
                        prob = calculate_beta(self.wind, neighbour.sector_type, direction)
                        rnd = random.random()
                        if rnd < prob:
                            neighbour.start_fire()
                            new_sectors_on_fire.append(neighbour)
                            # Add to update list if not already there
                            if neighbour not in sectors_to_update:
                                sectors_to_update.append(neighbour)

            self.sectors_on_fire = [s for s in self.all_sectors if s.fire_state is FireState.ACTIVE]
            
            if len(self.sectors_on_fire) == 0:
                if self._tick_count == 1 or random.random() < 0.25:
                    try:
                        row = self._map._rows // 2
                        column = self._map._columns // 2
                        target = self._map._sectors[row][column]
                        if target.fire_state is FireState.INACTIVE:
                            target.start_fire()
                            self.sectors_on_fire.append(target)
                            logger.info("Auto-ignited sector at row=%d, col=%d, id=%s", row, column, target.sector_id)
                    except Exception as e:
                        logger.warning("Auto-ignite failed: %s", e)

            self.wind.update_wind()
            
            # Use actual tick interval for agent movement, not fixed 0.1
            tick_delta = self.simulation_settings.tick_interval
            self.agents_manager.update(tick_delta)
            agent_states = self.agents_manager.get_agent_states()

            for sector in self.all_sectors:
                # update_sensors() returns jsons_by_type dict - use it directly to avoid duplicate iteration
                sector_sensor_data = sector.update_sensors()
                
                # Merge sector sensor data into output messages
                for sensor_type, sensor_list in sector_sensor_data.items():
                    if sensor_type not in out_messages:
                        out_messages[sensor_type] = []
                    out_messages[sensor_type].extend(sensor_list)
                
                # Only send sector state if it has been modified (performance optimization)
                if sector.is_modified:
                    sector_states.append(sector.make_sector_json())
                    sector.reset_modified_flag()

        # Only log summary every 10 ticks to reduce log volume
        if self._tick_count % 10 == 0:
            total_sensor_msgs = sum(len(v) for v in out_messages.values())
            logger.info(
                "Tick: %s, Active fires: %d, Sensor messages: %d, Sector states: %d",
                self._tick_count,
                len(self.sectors_on_fire),
                total_sensor_msgs,
                len(sector_states),
            )
        
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
            "config_loaded": self.config is not None,
            "speed_factor": getattr(self, 'speed_factor', 1.0)
        }

    def is_running(self) -> bool:
        return self._running
from engine.base import SimulationEngine 

from typing import Dict, Any

class SimpleSimulationEngine(SimulationEngine):
    def __init__(self):
        self.config: Optional[dict] = None
        self._map: Optional[ForestMap] = None
        self.agents_manager: Optional[AgentManager] = None
        self.wind: Optional[Wind] = None
        self.all_sectors: List[Sector] = []
        self.sectors_on_fire: List[Sector] = []
        self._running = False

    async def load_config(self, configuration: dict) -> None:
        self.config = configuration
        self._map = ForestMap.from_conf(configuration)
        self.agents_manager = AgentManager(self._map, None) 
        self.all_sectors = [s for row in self._map.sectors for s in row]

        first = self._map.start_new_fire()
        if first:
            self.sectors_on_fire = [first]
        else:
            self.sectors_on_fire = []
        self.wind = Wind()

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def step(self, ticks: int = 1) -> Dict[str, Any]:
        """
        Update Simulation step. Return dict:
        {
            "sensor_messages": { sensor_type_name: [json_obj, ...], ... },
            "sector_states": [json_obj, ...],
            "events": [...]
        }
        """

        out_messages: Dict[str, List[Any]] = {}
        sector_states: List[Any] = []
        events: List[Any] = []

        if not (self._map and self.wind and self.agents_manager):
            return {
                "sensor_messages": out_messages, 
                "sector_states": sector_states, 
                "events": events
            }

        for _ in range(ticks):
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

            self.sectors_on_fire = [s for s in self.sectors_on_fire if s.fire_state is FireState.ACTIVE]
            self.sectors_on_fire.extend(new_sectors_on_fire)

            self.wind.update_wind()

            for sector in self.all_sectors:
                sector.update_sensors()
                for sensor_type, jsons in sector.make_jsons().items():
                    out_messages.setdefault(sensor_type, []).extend(jsons)
                sector_states.append(sector.make_sector_json())

            self.agents_manager.update_agents_states()

        return {
            "sensor_messages": out_messages, 
            "sector_states": sector_states, 
            "events": events
        }

    async def pause(self) -> None:
        raise NotImplementedError("Pause functionality is not implemented yet.")

    async def load_config(self, config: dict) -> None:
        self._config = config

    def snapshot(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "sectors_on_fire": len(self.sectors_on_fire),
            "total_sectors": len(self.all_sectors),
            "config_loaded": self.config is not None
        }

    def is_running(self) -> bool:
        return self._running
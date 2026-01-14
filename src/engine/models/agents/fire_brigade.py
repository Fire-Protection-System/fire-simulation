import logging
from datetime import datetime
from typing import Optional

from src.engine.models.agents.agent import Agent
from src.engine.models.agents.agent_state import AgentState
from src.engine.models.core.location import Location
from src.engine.models.map.sector import Sector

logger = logging.getLogger(__name__)

class FireBrigade(Agent):
    """Fire brigade agent - runtime only (no decision logic)"""
    
    def __init__(
        self,
        fire_brigade_id: str,
        timestamp: datetime,
        base_location: Location,
        initial_location: Location,
        initial_state: Optional[AgentState] = None,
    ):
        super().__init__(fire_brigade_id, timestamp, base_location, initial_location)
        self._extinguishing_rate = 0.1  # Fire level reduction per second
        
        # Override state if provided (for backward compatibility)
        if initial_state:
            if isinstance(initial_state, AgentState):
                self._state = initial_state
            else:
                # Handle old AGENT_STATE enum
                from src.engine.models.agents.agent_state import AGENT_STATE
                if initial_state == AGENT_STATE.AVAILABLE:
                    self._state = AgentState.IDLE
                elif initial_state == AGENT_STATE.TRAVELLING:
                    self._state = AgentState.TRAVELING
                elif initial_state == AGENT_STATE.EXTINGUISHING:
                    self._state = AgentState.EXECUTING
    
    def execute_task(self, delta: float, sector: Optional[Sector]) -> bool:
        """Extinguish fire - returns True when sector is clear"""
        if not sector:
            return True
        
        if sector.fire_level <= 0:
            logger.debug(f"FireBrigade {self._agent_id}: sector already clear")
            return True
        
        # Reduce fire level based on extinguishing rate
        reduction = self._extinguishing_rate * delta
        sector.fire_level = max(0, sector.fire_level - reduction)
        
        logger.debug(f"FireBrigade {self._agent_id}: fire_level now {sector.fire_level:.2f}")
        
        return sector.fire_level <= 0
    
    def get_task_progress(self, sector: Optional[Sector]) -> float:
        """Return fire extinguishing progress (0.0 = full fire, 1.0 = extinguished)"""
        if not sector:
            return 1.0
        
        # Assume initial fire level was 1.0, progress is inverse of current level
        return 1.0 - min(1.0, sector.fire_level)
    
    def can_execute_task(self, sector: Sector) -> bool:
        """Can only extinguish if fire present"""
        return sector.fire_level > 0
    
    def increment_agents_in_sector(self, sector: Sector):
        """Track number of fire brigades in sector"""
        sector._number_of_fire_brigades += 1
    
    def decrement_agents_in_sector(self, sector: Sector):
        """Track number of fire brigades in sector"""
        sector._number_of_fire_brigades -= 1
    
    @property
    def fire_brigade_id(self) -> str:
        return self.agent_id
    
    # ===== BACKWARD COMPATIBILITY =====
    
    @property
    def initial_state(self) -> AgentState:
        """Deprecated: Use state instead"""
        return self._state
    
    def clone(self) -> 'FireBrigade':
        """Create a copy of this fire brigade"""
        return FireBrigade(
            fire_brigade_id=self._agent_id,
            timestamp=self._timestamp,
            base_location=Location(self._base_location.latitude, self._base_location.longitude),
            initial_location=Location(self._location.latitude, self._location.longitude),
            initial_state=self._state
        )

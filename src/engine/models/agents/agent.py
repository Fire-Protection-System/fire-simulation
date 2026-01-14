from abc import ABC, abstractmethod
from datetime import datetime
import logging
from typing import Optional, Dict, Any

from src.engine.models.agents.agent_state import AgentState, AGENT_STATE
from src.engine.models.agents.agent_perception import AgentPerception
from src.engine.models.core.location import Location
from src.engine.models.map.sector import Sector

logger = logging.getLogger(__name__)

class Agent(ABC):
    """
    Agent runtime - handles physics, movement, state transitions (NOT decisions).
    Decision-making is external (backend/support system).
    """
    def __init__(
        self,
        agent_id: str,
        timestamp: datetime,
        base_location: Location,
        initial_location: Location,
    ):
        self._agent_id = agent_id
        self._timestamp = timestamp
        self._base_location = base_location
        self._location = initial_location
        self._destination = base_location
        self._state = AgentState.IDLE
            
    def perceive(self, current_sector: Optional[Sector] = None) -> AgentPerception:
        """Gather information about current state (for decision-making)"""
        return AgentPerception(
            current_location=self._location,
            current_sector=current_sector,
            destination=self._destination,
            distance_to_destination=self._calculate_distance(self._location, self._destination),
            task_progress=self.get_task_progress(current_sector) if current_sector else 0.0,
            base_location=self._base_location
        )
        
    def update_physics(self, delta: float, map_ref) -> Dict[str, Any]:
        """Update agent position/state based on physics (called every simulation tick)"""
        if self._state == AgentState.TRAVELING or self._state == AgentState.RETURNING:
            old_location = (self._location.latitude, self._location.longitude)
            reached = self._move_towards_destination(delta)
            new_location = (self._location.latitude, self._location.longitude)
            current_sector = map_ref.find_sector(self._location) if hasattr(map_ref, 'find_sector') else None
            
            if old_location != new_location:
                logger.debug(f"[PHYSICS] Agent {self._agent_id} moved from ({old_location[0]:.6f}, {old_location[1]:.6f}) to ({new_location[0]:.6f}, {new_location[1]:.6f}), state: {self._state.value}")
            
            if reached:
                if self._destination == self._base_location:
                    self._state = AgentState.IDLE
                    logger.debug(f"[PHYSICS] Agent {self._agent_id} reached base, state changed to: {self._state.value}")
                    return {"event": "reached_base", "sector": current_sector}
                else:
                    self._state = AgentState.EXECUTING
                    logger.debug(f"[PHYSICS] Agent {self._agent_id} reached destination, state changed to: {self._state.value}")
                    if current_sector:
                        self.increment_agents_in_sector(current_sector)
                    return {"event": "reached_destination", "sector": current_sector}
            
            return {"event": "moving", "sector": current_sector}
            
        elif self._state == AgentState.EXECUTING:
            current_sector = map_ref.find_sector(self._location) if hasattr(map_ref, 'find_sector') else None
            task_complete = self.execute_task(delta, current_sector)
            if task_complete:
                # When task is complete, automatically return to base instead of staying idle on sector
                self._destination = self._base_location
                self._state = AgentState.RETURNING
                if current_sector:
                    self.decrement_agents_in_sector(current_sector)
                return {"event": "task_complete", "sector": current_sector}
        
        return {"event": "idle", "sector": None}
    
    def _move_towards_destination(self, delta: float) -> bool:
        """Update position, return True if reached destination"""
        # Movement speed: 0.005 degrees per second (independent of tick interval)
        # This ensures agents move at consistent speed regardless of simulation tick rate
        movement_speed = 0.005  # degrees per second
        movement_delta = movement_speed * delta
        
        step_lat = self._calculate_step(self._destination.latitude, self._location.latitude, movement_delta)
        step_lon = self._calculate_step(self._destination.longitude, self._location.longitude, movement_delta)
        
        self._location.latitude += step_lat
        self._location.longitude += step_lon
        
        dist = self._calculate_distance(self._location, self._destination)
        return dist <= 0.0001
    
    def _calculate_step(self, target: float, current: float, delta: float) -> float:
        """Calculate movement step towards target"""
        if target > current:
            return min(delta, target - current)
        elif target < current:
            return max(-delta, target - current)
        return 0.0
    
    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """Calculate Euclidean distance between two locations"""
        return ((loc1.latitude - loc2.latitude)**2 + (loc1.longitude - loc2.longitude)**2)**0.5
        
    def execute_command(self, command: Dict[str, Any]):
        """Execute external command (from backend/support)"""
        cmd_type = command.get("type")
        old_state = self._state.value
        
        logger.debug(f"[AGENT] Agent {self._agent_id} received command: {cmd_type}")
        
        if cmd_type == "move_to":
            loc_data = command.get("location", {})
            if not loc_data:
                logger.error(f"[AGENT] Agent {self._agent_id} move_to command missing location: {command}")
                return
            target = Location(loc_data.get("latitude"), loc_data.get("longitude"))
            self._destination = target
            self._state = AgentState.TRAVELING
            logger.debug(f"[AGENT] Agent {self._agent_id} state changed: {old_state} -> {self._state.value}, destination: ({target.latitude:.6f}, {target.longitude:.6f})")
            
        elif cmd_type == "return_to_base":
            self._destination = self._base_location
            self._state = AgentState.RETURNING
            logger.debug(f"[AGENT] Agent {self._agent_id} state changed: {old_state} -> {self._state.value}, returning to base")
            
        elif cmd_type == "abort":
            self._state = AgentState.IDLE
            logger.debug(f"[AGENT] Agent {self._agent_id} state changed: {old_state} -> {self._state.value}, task aborted")
        else:
            logger.warning(f"[AGENT] Agent {self._agent_id} received unknown command type: {cmd_type}")
        
    @abstractmethod
    def execute_task(self, delta: float, sector: Optional[Sector]) -> bool:
        """Execute agent-specific task, return True when complete"""
        pass
    
    @abstractmethod
    def get_task_progress(self, sector: Optional[Sector]) -> float:
        """Return task completion (0.0 - 1.0)"""
        pass
    
    @abstractmethod
    def can_execute_task(self, sector: Sector) -> bool:
        """Check if agent can perform task in given sector"""
        pass
    
    @abstractmethod
    def increment_agents_in_sector(self, sector: Sector):
        """Increment agent counter in sector (for tracking)"""
        pass
    
    @abstractmethod
    def decrement_agents_in_sector(self, sector: Sector):
        """Decrement agent counter in sector (for tracking)"""
        pass
        
    @property
    def agent_id(self) -> str:
        return self._agent_id
    
    @property
    def state(self) -> AgentState:
        return self._state
    
    @property
    def location(self) -> Location:
        return self._location
    
    @property
    def destination(self) -> Location:
        return self._destination
    
    @property
    def base_location(self) -> Location:
        return self._base_location
    
    @property
    def timestamp(self) -> datetime:
        return self._timestamp
        

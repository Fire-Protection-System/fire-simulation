"""
Contracts/Interfaces for fire-simulation message formats.

This module defines the expected structure of all messages sent to queues.
These contracts ensure consistency between the simulation engine and consumers (frontend, backend, support services).
"""

from typing import Dict, Any, Optional, TypedDict
from dataclasses import dataclass


# ============================================================================
# SECTOR STATE CONTRACT
# ============================================================================

@dataclass
class SectorStateMessage:
    """
    Contract for sector state update messages sent to simulation.telemetry.map.sector_state topic.
    
    This is the MINIMAL data needed by the frontend to update a sector's fire-related state.
    Only changed sectors should be sent, not all sectors every tick.
    """
    sectorId: int
    fireLevel: float
    burnLevel: float
    extinguishLevel: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sectorId": self.sectorId,
            "fireLevel": self.fireLevel,
            "burnLevel": self.burnLevel,
            "extinguishLevel": self.extinguishLevel
        }


# ============================================================================
# SENSOR MESSAGE CONTRACT
# ============================================================================

@dataclass
class SensorMessage:
    """
    Base contract for sensor telemetry messages.
    All sensor messages must include these fields.
    """
    sensorId: int
    timestamp: str  # ISO format: YYYY-MM-DDTHH:MM:SS
    sensorType: str
    location: Dict[str, float]  # {"longitude": float, "latitude": float}
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sensorId": self.sensorId,
            "timestamp": self.timestamp,
            "sensorType": self.sensorType,
            "location": self.location,
            "data": self.data
        }


# ============================================================================
# AGENT STATE CONTRACT
# ============================================================================

@dataclass
class AgentStateMessage:
    """
    Contract for agent (fire brigade / forester patrol) state messages.
    Sent to simulation.telemetry.agents.fire_brigade or simulation.telemetry.agents.forester topics.
    """
    timestamp: str  # ISO format
    event: str  # "idle", "traveling", "executing", "reached_destination", "reached_base", etc.
    location: Dict[str, float]  # {"latitude": float, "longitude": float}
    sectorId: Optional[int]
    destination: Dict[str, float]  # {"latitude": float, "longitude": float}
    state: str  # "AVAILABLE", "TRAVELLING", "EXTINGUISHING", "PATROLLING"
    type: str  # "fire_brigade" or "forester_patrol"
    # ID field name varies by agent type (fireBrigadeId or foresterPatrolId)
    # Value stored in agent_id field
    
    def to_dict(self, id_field_name: str, agent_id: int) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "timestamp": self.timestamp,
            "event": self.event,
            "location": self.location,
            "sectorId": self.sectorId,
            "destination": self.destination,
            "state": self.state,
            "type": self.type
        }
        result[id_field_name] = agent_id
        return result


# ============================================================================
# COMMAND CONTRACT (for receiving commands)
# ============================================================================

@dataclass
class AgentCommand:
    """
    Contract for agent commands received from backend/support services.
    Received via simulation.control.fire_brigade_actions or simulation.control.forester_actions topics.
    """
    type: str  # "move_to", "return_to_base", "abort"
    agentId: str  # Format: "FB-{id}" for fire brigades, "FP-{id}" for forester patrols
    location: Optional[Dict[str, float]] = None  # Required for "move_to" commands
    # Legacy format also supported: fireBrigadeId/foresterPatrolId + goingToBase flag


# ============================================================================
# EVENT CONTRACT
# ============================================================================

@dataclass
class SimulationEventMessage:
    """
    Contract for simulation event messages.
    Sent to simulation.events topic.
    """
    timestamp: str
    eventType: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "eventType": self.eventType,
            "data": self.data
        }

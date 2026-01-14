import logging
from typing import Dict, List, Optional
from datetime import datetime

from src.engine.models.agents.agent import Agent
from src.engine.models.agents.fire_brigade import FireBrigade
from src.engine.models.agents.forester_patrol import ForesterPatrol
from src.engine.models.map.forest_map import ForestMap
from src.engine.models.map.sector import Sector
from src.rabbitmq.message_store import MessageStore
from src.messaging.topics import TopicRegistry
from src.engine.agent_manager.agent_type_config import get_agent_config

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Agent orchestration layer - manages agent lifecycle, commands, telemetry.
    Does NOT make decisions - only executes external commands.
    """
    
    def __init__(self, 
            forest_map: ForestMap, 
            message_store: MessageStore = None, 
            engine                      = None
        ):
        self._map = forest_map
        self._message_store = message_store
        self._engine = engine
        self._agents: Dict[str, Agent] = {}
        self._agent_sectors: Dict[str, Optional[Sector]] = {}
        self._brigades: Dict[str, FireBrigade] = {}
        self._patrols: Dict[str, ForesterPatrol] = {}
        
        # Initialize agents from map
        # Use unique prefixes to avoid ID collisions between fire brigades and forester patrols
        for brigade in forest_map.fire_brigades:
            agent_id = f"FB-{brigade.fire_brigade_id}"  # Fire Brigade prefix
            self._agents[agent_id] = brigade
            # Also store with numeric ID for backward compatibility
            numeric_id = str(brigade.fire_brigade_id)
            self._brigades[numeric_id] = brigade
            self._agent_sectors[agent_id] = forest_map.find_sector(brigade.location)
        
        for patrol in forest_map.forester_patrols:
            agent_id = f"FP-{patrol.forester_patrol_id}"  # Forester Patrol prefix
            self._agents[agent_id] = patrol
            # Also store with numeric ID for backward compatibility
            numeric_id = str(patrol.forester_patrol_id)
            self._patrols[numeric_id] = patrol
            self._agent_sectors[agent_id] = forest_map.find_sector(patrol.location)
        
        logger.info(f"AgentManager initialized with {len(self._agents)} agents "
                   f"({len(self._brigades)} brigades, {len(self._patrols)} patrols)")
    
    # ===== SIMULATION UPDATE =====
    
    def update(self, delta_time: float):
        """
        Called every simulation tick - updates all agent physics/state.
        Uses new update_physics method from refactored Agent class.
        """
        speed_factor = getattr(self._engine, 'speed_factor', 1.0)
        adjusted_delta = delta_time * speed_factor
        
        for agent_id, agent in self._agents.items():
            old_state = agent.state.value
            event = agent.update_physics(adjusted_delta, self._map)
            new_state = agent.state.value
            
            # Update sector tracking
            current_sector = self._map.find_sector(agent.location)
            self._agent_sectors[agent_id] = current_sector
            
            # Publish telemetry based on event
            self._publish_telemetry(agent, event, current_sector)
    
    # ===== COMMAND PROCESSING =====
    
    def process_command(self, command: dict):
        """
        Process external command (from backend/support via CommandConsumer).
        
        Expected command format:
        {
            "type": "move_to" | "return_to_base" | "abort",
            "agentId": "FB-01" | "FP-01",
            "location": {"latitude": 52.0, "longitude": 21.0}  # for move_to
        }
        """
        agent_id = str(command.get("agentId", ""))
        
        if not agent_id:
            logger.warning(f"Command missing agent ID: {command}")
            return
        
        agent = self._agents.get(agent_id)
        if not agent:
            logger.error(f"[COMMAND] Agent not found: {agent_id}. Available agents: {list(self._agents.keys())}")
            return
        
        try:
            agent.execute_command(command)
        except Exception as e:
            logger.error(f"[COMMAND] Failed to execute command for {agent_id}: {e}", exc_info=True)
    
    # ===== TELEMETRY =====
    
    def _publish_telemetry(self, agent: Agent, event: dict, sector: Optional[Sector]):
        """Publish agent state to RabbitMQ (configuration-driven)"""
        if not self._message_store:
            return
        
        config = get_agent_config(agent)
        event_type = event.get("event", "idle")
        
        # Get agent ID based on type
        if isinstance(agent, FireBrigade):
            agent_id_value = agent.fire_brigade_id
        elif isinstance(agent, ForesterPatrol):
            agent_id_value = agent.forester_patrol_id
        else:
            agent_id_value = agent.agent_id
        
        # Map simulator state to backend enum values
        state_value = agent.state.value
        if isinstance(agent, FireBrigade):
            # Map: idle -> AVAILABLE, traveling -> TRAVELLING, executing -> EXTINGUISHING, returning -> TRAVELLING
            state_mapping = {
                "idle": "AVAILABLE",
                "traveling": "TRAVELLING",
                "executing": "EXTINGUISHING",
                "returning": "TRAVELLING"
            }
            state_value = state_mapping.get(state_value, "AVAILABLE")
        elif isinstance(agent, ForesterPatrol):
            # Map: idle -> AVAILABLE, traveling -> TRAVELLING, executing -> PATROLLING, returning -> TRAVELLING
            state_mapping = {
                "idle": "AVAILABLE",
                "traveling": "TRAVELLING",
                "executing": "PATROLLING",
                "returning": "TRAVELLING"
            }
            state_value = state_mapping.get(state_value, "AVAILABLE")
        
        message = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "location": {
                "latitude": agent.location.latitude,
                "longitude": agent.location.longitude
            },
            "sectorId": sector.sector_id if sector else None,
            "destination": {
                "latitude": agent.destination.latitude,
                "longitude": agent.destination.longitude
            },
            "state": state_value,
            config.id_field_name: agent_id_value,
            "type": config.type_name
        }
        
        self._message_store.add_message_to_sent(config.telemetry_topic, message)
        
        # Removed verbose event logging
    
    # ===== STATE QUERIES =====
    
    def get_agent_states(self) -> List[dict]:
        """
        Return current state of all agents as JSON objects matching backend format.
        Uses configuration-driven approach instead of isinstance() checks.
        """
        states = []
        timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        
        for agent_id, agent in self._agents.items():
            sector = self._agent_sectors.get(agent_id)
            config = get_agent_config(agent)
            
            # Map simulator state to backend enum values
            state_value = agent.state.value
            if isinstance(agent, FireBrigade):
                # Map: idle -> AVAILABLE, traveling -> TRAVELLING, executing -> EXTINGUISHING, returning -> TRAVELLING
                state_mapping = {
                    "idle": "AVAILABLE",
                    "traveling": "TRAVELLING",
                    "executing": "EXTINGUISHING",
                    "returning": "TRAVELLING"
                }
                state_value = state_mapping.get(state_value, "AVAILABLE")
            elif isinstance(agent, ForesterPatrol):
                # Map: idle -> AVAILABLE, traveling -> TRAVELLING, executing -> PATROLLING, returning -> TRAVELLING
                state_mapping = {
                    "idle": "AVAILABLE",
                    "traveling": "TRAVELLING",
                    "executing": "PATROLLING",
                    "returning": "TRAVELLING"
                }
                state_value = state_mapping.get(state_value, "AVAILABLE")
            
            state_dict = {
                "timestamp": timestamp,
                "state": state_value,
                "location": {
                    "latitude": agent.location.latitude,
                    "longitude": agent.location.longitude
                },
                "sectorId": sector.sector_id if sector else None,
                "destination": {
                    "latitude": agent.destination.latitude,
                    "longitude": agent.destination.longitude
                },
                "baseLocation": {
                    "latitude": agent.base_location.latitude,
                    "longitude": agent.base_location.longitude
                },
                "type": config.type_name,
                config.id_field_name: agent.fire_brigade_id if isinstance(agent, FireBrigade) else (agent.forester_patrol_id if isinstance(agent, ForesterPatrol) else agent.agent_id)
            }
            
            states.append(state_dict)
        
        return states
    

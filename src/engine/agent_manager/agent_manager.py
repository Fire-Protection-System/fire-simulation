import logging
import os
from typing import Dict, List, Optional
from datetime import datetime
import asyncio

from src.engine.models.agents.agent import Agent
from src.engine.models.agents.fire_brigade import FireBrigade
from src.engine.models.agents.forester_patrol import ForesterPatrol
from src.engine.models.map.forest_map import ForestMap
from src.engine.models.map.sector import Sector
from src.rabbitmq.message_store import MessageStore
from src.messaging.topics import TopicRegistry
from src.engine.agent_manager.agent_type_config import get_agent_config

logger = logging.getLogger(__name__)

_telemetry_count = 0
_last_telemetry_log = None

class AgentManager:
    """
    Agent orchestration layer - manages agent lifecycle, commands, telemetry.
    Does NOT make decisions - only executes external commands.
    """
    def __init__(
        self, 
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
        self._llm_brains: Dict[str, any] = {} 
        self._agent_communication = None
        self._enable_llm_agents = os.environ.get("ENABLE_LLM_AGENTS", "true").lower() == "true"
        self._telemetry_batch: Dict[str, List[dict]] = {}
        self._last_telemetry_flush_ts: float = 0.0
        
        for brigade in forest_map.fire_brigades:
            agent_id = f"FB-{brigade.fire_brigade_id}"
            self._agents[agent_id] = brigade
            numeric_id = str(brigade.fire_brigade_id)
            self._brigades[numeric_id] = brigade
            self._agent_sectors[agent_id] = forest_map.find_sector(brigade.location)
            
        
        for patrol in forest_map.forester_patrols:
            agent_id = f"FP-{patrol.forester_patrol_id}"
            self._agents[agent_id] = patrol
            numeric_id = str(patrol.forester_patrol_id)
            self._patrols[numeric_id] = patrol
            self._agent_sectors[agent_id] = forest_map.find_sector(patrol.location)
        
        if self._enable_llm_agents and self._message_store:
            try:
                from src.llm.agent_communication import AgentCommunication
                self._agent_communication = AgentCommunication(self._message_store)
                logger.debug("[LLM] Agent communication system initialized")
                
                for agent in self._agents.values():
                    agent.set_communication(self._agent_communication)
                logger.debug(f"[LLM] Assigned communication adapter to {len(self._agents)} agents")

            except Exception as e:
                logger.warning(f"[LLM] Failed to initialize agent communication: {e}")
        
        if self._agent_communication:
            for agent in self._agents.values():
                if hasattr(agent, '_llm_chat_enabled'):
                    agent._llm_chat_enabled = True
                    agent.set_communication(self._agent_communication)
        
        logger.info(f"AgentManager initialized with {len(self._agents)} agents")

    def update(self, delta_time: float, publish_telemetry: bool = True):
        speed_factor = getattr(self._engine, 'speed_factor', 1.0)
        adjusted_delta = delta_time * speed_factor

        for agent_id, agent in list(self._agents.items()):
            old_state = agent.state.value
            
            if publish_telemetry and self._enable_llm_agents and isinstance(agent, FireBrigade) and self._agent_communication:
                self._announce_agent_state_changes(agent_id, agent, old_state)
            
            '''
            Update agent physics and state
            Update state tasks 
            '''
            event = agent.update_physics(adjusted_delta, self._map)
            new_state = agent.state.value
            
            if publish_telemetry:
                current_sector = self._map.find_sector(agent.location)
                self._agent_sectors[agent_id] = current_sector
                self._publish_telemetry(agent, event, current_sector)
    
    def process_command(self, command: dict):
        agent_id = str(command.get("agentId", ""))
        if not agent_id: return
        
        agent = self._agents.get(agent_id)
        if not agent:
            logger.error(f"[COMMAND] Agent not found: {agent_id}")
            return
        
        try:
            agent.execute_command(command)
        except Exception as e:
            logger.error(f"[COMMAND] Failed to execute command for {agent_id}: {e}")

    def _publish_telemetry(self, agent: Agent, event: dict, sector: Optional[Sector]):
        if not self._message_store: return
        config = get_agent_config(agent)
        agent_id_val = agent.fire_brigade_id if isinstance(agent, FireBrigade) else (agent.forester_patrol_id if isinstance(agent, ForesterPatrol) else agent.agent_id)

        try:
            if isinstance(agent_id_val, str) and agent_id_val.isdigit():
                agent_id_val = int(agent_id_val)
        except Exception:
            pass
        
        s_map = {"idle": "AVAILABLE", "traveling": "TRAVELLING", "executing": ("EXTINGUISHING" if isinstance(agent, FireBrigade) else "PATROLLING"), "returning": "TRAVELLING"}
        state_val = s_map.get(agent.state.value, "AVAILABLE")
        session_id = getattr(self._engine, '_simulation_session_id', None)
        
        message = {
            "timestamp": datetime.now().isoformat(),
            "event" : event.get("event", "idle"),
            "location": {"latitude": agent.location.latitude, "longitude": agent.location.longitude},
            "sectorId": sector.sector_id if sector else None,
            "destination": {"latitude": agent.destination.latitude, "longitude": agent.destination.longitude},
            "state": state_val,
            config.id_field_name: agent_id_val,
            "type": config.type_name,
            "simulationSessionId": session_id
        }

        self.queue_telemetry(config.telemetry_topic, message)

    def queue_telemetry(self, topic: str, message: dict):
        """Accumulate messages per topic and flush later."""
        if not topic: return
        lst = self._telemetry_batch.setdefault(topic, [])
        lst.append(message)

    def flush_telemetry(self):
        """Send batched telemetry messages once per tick."""
        if not self._message_store:
            self._telemetry_batch.clear()
            return
        from datetime import datetime
        for topic, messages in list(self._telemetry_batch.items()):
            if not messages:
                continue
            batch_msg = {
                "timestamp": datetime.now().isoformat(),
                "batch": messages
            }
            try:
                self._message_store.add_message_to_sent(topic, batch_msg)
            except Exception:
                # swallow to avoid interrupting simulation
                pass
        self._telemetry_batch.clear()

    def get_agent_states(self) -> List[dict]:
        states = []
        ts = datetime.now().isoformat()
        for agent_id, agent in self._agents.items():
            sector = self._agent_sectors.get(agent_id)
            config = get_agent_config(agent)

            s_map = {
                "idle"     : "AVAILABLE", 
                "traveling": "TRAVELLING", 
                "executing": ("EXTINGUISHING" if isinstance(agent, FireBrigade) else "PATROLLING"), 
                "returning": "TRAVELLING"}
            
            destination = {
                "latitude":  agent.destination.latitude, 
                "longitude": agent.destination.longitude
            } if agent.destination else None

            location = {
                "latitude": agent.location.latitude, 
                "longitude": agent.location.longitude
            } if agent.location else None

            baseLocation = {
                "latitude": agent.base_location.latitude, 
                "longitude": agent.base_location.longitude
            } if agent.base_location else None


            # Determine the actual ID value for this agent (not the field name)
            if isinstance(agent, FireBrigade):
                id_value = agent.fire_brigade_id
            elif isinstance(agent, ForesterPatrol):
                id_value = agent.forester_patrol_id
            else:
                id_value = agent.agent_id

            # Coerce numeric ID strings to integers to match backend types when possible
            try:
                if isinstance(id_value, str) and id_value.isdigit():
                    id_value = int(id_value)
            except Exception:
                pass

            states.append({
                "timestamp"          : ts,
                "state"              : s_map.get(agent.state.value, "AVAILABLE"),
                "location"           : location,
                "sectorId"           : sector.sector_id if sector else None,
                "destination"        : destination,
                "baseLocation"       : baseLocation,
                "type"               : config.type_name,
                config.id_field_name : id_value
            })
        return states

    def _announce_agent_state_changes(self, agent_id: str, agent: FireBrigade, old_state: str):
        if not self._agent_communication: return
        curr = agent.state.value

        match (old_state, curr):
            case ("idle", "idle"):
                return
            case ('idle', 'traveling'):
                target = self._map.find_sector(agent.destination)
                self._announce_action(agent_id, "order_received", target.sector_id if target else None, "Moving to task")
            case ('traveling', 'executing'):
                sector = self._agent_sectors.get(agent_id)
                self._announce_action(agent_id, "starting_extinguish", sector.sector_id if sector else None)
            case ('executing', 'returning'):
                sector = self._agent_sectors.get(agent_id)
                self._announce_action(agent_id, "task_complete", sector.sector_id if sector else None)
            case _:
                logger.debug(f"[LLM] No announcement rule for state change {old_state} -> {curr} for agent {agent_id}")



    def _announce_action(self, agent_id: str, action: str, target_sector_id: Optional[int] = None, reasoning: Optional[str] = None):
        if not self._agent_communication: return
        agent = self._agents.get(agent_id)
        if agent:
            self._agent_communication.announce_action(agent_id, action, target_sector_id, {"latitude": agent.location.latitude, "longitude": agent.location.longitude}, reasoning)

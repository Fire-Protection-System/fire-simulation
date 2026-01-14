import logging
import threading
import time
from typing import Callable, Optional

from src.rabbitmq.message_store import MessageStore
from src.messaging.topics import TopicRegistry
from src.engine.agent_manager.agent_type_config import get_all_agent_configs, AgentTypeConfig

logger = logging.getLogger(__name__)

class CommandConsumer:
    """
    Consumes commands from RabbitMQ and forwards to AgentManager.
    Runs in background thread to continuously process incoming commands.
    """
    
    def __init__(self, message_store: MessageStore, command_callback: Callable):
        """
        Initialize command consumer.
        
        Args:
            message_store: MessageStore instance for RabbitMQ communication
            command_callback: Function to call with processed commands (e.g., agent_manager.process_command)
        """
        self._message_store = message_store
        self._command_callback = command_callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start consuming commands in background thread"""
        if self._running:
            logger.warning("CommandConsumer already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True, name="CommandConsumer")
        self._thread.start()
        logger.info("CommandConsumer started")
    
    def stop(self):
        """Stop consuming commands"""
        if not self._running:
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("CommandConsumer thread did not stop gracefully")
        logger.info("CommandConsumer stopped")
    
    def _consume_loop(self):
        """Main consumption loop - configuration-driven approach"""
        logger.info("CommandConsumer loop started")
        agent_configs = get_all_agent_configs()
        
        while self._running:
            try:
                for config in agent_configs:
                    # Convert topic (with dots) to queue name (with underscores) for message retrieval
                    # All messages (from RabbitMQ and REST API) are now stored with queue names
                    queue_name = config.command_topic.replace('.', '_')
                    message = self._message_store.get_received_message(queue_name)
                    if message:
                        self._process_agent_command(message, config)
                
            except Exception as e:
                logger.error(f"Error in command consumer loop: {e}", exc_info=True)
            
            time.sleep(0.1)
        
        logger.info("CommandConsumer loop exited")
    
    def _process_agent_command(self, message: dict, config: AgentTypeConfig):
        """
        Generic agent command processor - works for ANY agent type.
        Replaces type-specific methods (_process_fire_brigade_command, _process_forester_command).
        
        Expected message formats (supports both):
        1. HTTP format:
           {
               "<id_field_name>": "AGENT-01",
               "goingToBase": false,
               "location": {"latitude": 52.0, "longitude": 21.0}
           }
        2. RabbitMQ OrderFireBrigade format:
           {
               "<id_field_name>": 0,
               "action": "GO_TO_BASE" | "EXTINGUISH",
               "location": {"latitude": 52.0, "longitude": 21.0},
               "timestamp": "...",
               "fireState": null
           }
        
        Args:
            message: RabbitMQ message dict
            config: AgentTypeConfig for this agent type
        """
        try:
            agent_id = message.get(config.id_field_name)
            if not agent_id:
                logger.warning(f"[CONSUMER] {config.display_name} command missing {config.id_field_name}: {message}")
                return
            
            # Convert agent_id to string if it's a number (from OrderFireBrigade format)
            agent_id = str(agent_id)
            
            # Add prefix to avoid ID collision between fire brigades and forester patrols
            # Fire brigades use "FB-{id}", forester patrols use "FP-{id}"
            if config.id_field_name == "fireBrigadeId":
                agent_id = f"FB-{agent_id}"
            elif config.id_field_name == "foresterPatrolId":
                agent_id = f"FP-{agent_id}"
            
            # Check for goingToBase (HTTP format) or action (RabbitMQ OrderFireBrigade format)
            going_to_base = message.get("goingToBase", False)
            action = message.get("action")
            
            # Handle OrderFireBrigade format (action enum)
            if action is not None:
                if action == "GO_TO_BASE":
                    going_to_base = True
                elif action == "EXTINGUISH":
                    going_to_base = False
                else:
                    logger.warning(f"{config.display_name} command has unknown action: {action}")
                    return
            
            if going_to_base:
                command = {
                    "type": "return_to_base",
                    "agentId": agent_id,
                }
            else:
                loc = message.get("location")
                if not loc:
                    logger.warning(f"[CONSUMER] {config.display_name} command missing location: {message}")
                    return
                
                command = {
                    "type": "move_to",
                    "agentId": agent_id,
                    "location": {
                        "latitude": loc.get("latitude"),
                        "longitude": loc.get("longitude")
                    }
                }
            
            self._command_callback(command)
            
        except Exception as e:
            logger.error(f"Failed to process {config.display_name} command: {e}", exc_info=True)

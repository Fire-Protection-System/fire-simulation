import asyncio
import threading
import time
import logging

from typing import Any, Dict

from engine.simple_simulation_engine import SimpleSimulationEngine

from src.settings.communucation_settings import CommunicationSettings
from src.settings.simulation_settings import SimulationSettings
from src.engine.base import SimulationEngine
from src.engine.models.sensors.sensor_type import SensorType
from src.messaging.topics import (
    TopicRegistry, 
    get_topic_for_sensor, 
    get_all_topics,
    SimulationTopics,
    ControlTopics
)
from src.rabbitmq.message_store import MessageStore
from src.rabbitmq.pika_client import PikaClient
from src.rabbitmq import producer, consumer, connection_manager

logger = logging.getLogger(__name__)

class EngineRunner:
    def __init__(
        self, 
        engine: SimulationEngine                = SimpleSimulationEngine(), 
        settings: CommunicationSettings         = CommunicationSettings(), 
        simulation_settings: SimulationSettings = SimulationSettings(),
        store: MessageStore                     = MessageStore(),
        client: PikaClient                      = PikaClient()
    ):
        '''
            Initialize the EngineRunner with the given simulation engine and settings.
        '''    
        self.engine: SimulationEngine   = engine
        self.settings                   = settings
        self.simulation_settings        = simulation_settings 
        self.store                      = store
        self.client                     = client

        # Internal control. The idea was to make it possible to stop the runner cleanly
        # Not fully implemented yet though, sorry
        self._stop                                 = threading.Event()
        self._loop_thread: threading.Thread | None = None
        self._write_threads                        = []
        self._read_threads                         = []
        self._tick_interval                        = simulation_settings.tick_interval
        self._base_tick_interval                   = simulation_settings.tick_interval
        self._min_tick_interval                    = 0.5  # Fastest tick: 0.5 seconds
        self._max_tick_interval                    = 10.0  # Slowest tick: 10 seconds
        self._command_consumer                     = None
        self._original_config                      = None
        self._last_fire_count                       = 0

    def set_tick_interval(self, seconds: float) -> None:
        """
        Update simulation tick interval (seconds between engine steps).

        This controls how often `_do_step_and_send` is executed.
        """
        try:
            value = float(seconds)
        except (TypeError, ValueError):
            logger.warning("Invalid tick interval value %r – keeping previous %s", seconds, self._tick_interval)
            return

        if value <= 0:
            logger.warning("Attempted to set non-positive tick interval %s – keeping previous %s", value, self._tick_interval)
            return

        logger.info("Updating EngineRunner tick interval from %s to %s seconds", self._tick_interval, value)
        self._tick_interval = value

    async def start(self, config: Dict[str, Any]) -> None:
        # Store original config for state restoration
        self._original_config = config.copy() if config else None
        
        await self.engine.load_config(config)
        
        if hasattr(self.engine, 'agents_manager') and self.engine.agents_manager:
            self.engine.agents_manager._message_store = self.store
            logger.info("Message store set for agent manager")
        
        await self.engine.start()
        self._setup_queues()
        
        if hasattr(self.engine, 'agents_manager') and self.engine.agents_manager:
            from src.engine.agent_manager.command_consumer import CommandConsumer
            self._command_consumer = CommandConsumer(
                message_store=self.store,
                command_callback=self.engine.agents_manager.process_command
            )
            self._command_consumer.start()
            logger.info("CommandConsumer started for processing agent orders")

        logger.info("Waiting 2.5 seconds for queues to initialize...")
        await asyncio.sleep(2.5)

        def loop():
            while not self._stop.is_set():
                try:
                    self._do_step_and_send()
                    # Adaptive tick rate: faster when fires are active
                    self._adjust_tick_interval()
                except Exception as e:
                    logger.exception(f"Error in simulation loop: {e}")
                time.sleep(self._tick_interval)

        self._loop_thread = threading.Thread(target=loop, daemon=True)
        self._loop_thread.start()
        logger.info("Simulation loop started")

    def _setup_queues(self):
        with self.client.connection_ctx() as (connection, channel):
            if connection and channel:
                try:
                    channel.exchange_declare(
                        exchange      = self.settings.exchange_name,
                        exchange_type = 'topic',
                        durable       = False
                    )
                    logger.info(f"Exchange '{self.settings.exchange_name}' ready")
                    
                    # Use new topics API instead of deprecated TOPIC_NAMES/QUEUE_NAMES
                    all_topics = get_all_topics()
                    for topic in all_topics:
                        # Queue name = topic name (simplified, no separate queue names)
                        queue_name = topic.replace('.', '_')
                        
                        channel.queue_declare(queue=queue_name, durable=False)
                        channel.queue_bind(
                            exchange    = self.settings.exchange_name,
                            queue       = queue_name,
                            routing_key = topic
                        )
                        logger.debug(f"Queue '{queue_name}' bound to topic '{topic}'")
                    
                    logger.info(f"Created and bound {len(all_topics)} queues")
                except Exception as e:
                    logger.error(f"Error setting up queues: {e}", exc_info=True)

        # Only start producer threads for topics the simulation actually publishes
        # Use SimulationTopics.ALL instead of all TopicRegistry topics
        for topic_value in SimulationTopics.ALL:
            # Find the TopicRegistry enum for this topic value
            topic_enum = next((t for t in TopicRegistry if t.value == topic_value), None)
            if topic_enum:
                self._start_producer_thread(topic_enum)

        for topic in [TopicRegistry.FORESTER_ACTIONS, TopicRegistry.FIRE_BRIGADE_ACTIONS]:
            self._start_consumer_thread(topic)

    def _start_producer_thread(self, topic):
        """Start a producer thread for the given topic."""
        thread = threading.Thread(
            target=producer.start_producing_messages,
            kwargs={
                "exchange": self.settings.exchange_name,
                "routing_key": topic.value,
                "store": self.store,
                "username": self.settings.rabbitmq_username,
                "password": self.settings.rabbitmq_password,
                "stop_event": self._stop
            },
            daemon=True
        )
        thread.start()
        self._write_threads.append(thread)
        logger.info(f"Started producer thread for '{topic.value}'")

    def _start_consumer_thread(self, topic):
        """Start a consumer thread for the given topic."""
        # Convert routing key (with dots) to queue name (with underscores)
        queue_name = topic.value.replace('.', '_')
        thread = threading.Thread(
            target=consumer.consume_messages_from_queue,
            kwargs={
                "queue_name": queue_name,
                "store": self.store,
                "username": self.settings.rabbitmq_username,
                "password": self.settings.rabbitmq_password,
                "stop_event": self._stop
            },
            daemon=True
        )
        thread.start()
        self._read_threads.append(thread)
        logger.info(f"Started consumer thread for routing key '{topic.value}' (queue: '{queue_name}')")

    def _adjust_tick_interval(self):
        """
        Adaptively adjust tick interval based on simulation state.
        Faster ticks when fires are active, slower when stable.
        """
        if not hasattr(self.engine, 'sectors_on_fire'):
            return
        
        current_fire_count = len(self.engine.sectors_on_fire)
        
        # If fire count changed significantly, adjust tick rate
        if abs(current_fire_count - self._last_fire_count) > 2:
            if current_fire_count > 0:
                # Active fires: use faster tick rate
                # Scale: more fires = faster ticks (but not too fast)
                # Formula: base * (1 - min(fires/20, 0.6))
                fire_factor = min(current_fire_count / 20.0, 0.6)
                new_interval = self._base_tick_interval * (1.0 - fire_factor)
                new_interval = max(new_interval, self._min_tick_interval)
            else:
                # No fires: use slower tick rate to save resources
                new_interval = min(self._base_tick_interval * 1.5, self._max_tick_interval)
            
            if abs(new_interval - self._tick_interval) > 0.5:  # Only log significant changes
                logger.debug(f"Adaptive tick interval: {self._tick_interval:.2f}s -> {new_interval:.2f}s (fires: {current_fire_count})")
                self._tick_interval = new_interval
        
        self._last_fire_count = current_fire_count
    
    def _do_step_and_send(self):
        result = self.engine.step(1)
        
        sector_states = result.get("sector_states", [])
        sensor_messages = result.get("sensor_messages", {})
        agent_states = result.get("agent_states", [])
        events = result.get("events", [])
        
        # Removed verbose logging - only log errors
        
        self._process_sensor_messages(sensor_messages)
        self._process_sector_states(sector_states)
        self._process_agent_states(agent_states)
        self._process_events(events)

    async def stop(self):
        logger.info("Stopping simulation runner...")
        self._stop.set()
        
        # Stop CommandConsumer if it exists
        if self._command_consumer:
            logger.info("Stopping CommandConsumer...")
            self._command_consumer.stop()
            self._command_consumer = None
        
        # Stop main simulation loop
        if self._loop_thread and self._loop_thread.is_alive():
            logger.info("Waiting for simulation loop to stop...")
            self._loop_thread.join(timeout=2)
            if self._loop_thread.is_alive():
                logger.warning("Simulation loop thread did not stop within timeout")
        
        # Stop producer and consumer threads
        logger.info("Stopping producer and consumer threads...")
        for t in self._write_threads + self._read_threads:
            if t.is_alive():
                t.join(timeout=1)
                if t.is_alive():
                    logger.warning(f"Thread {t.name} did not stop within timeout")
        
        # Stop engine
        logger.info("Stopping engine...")
        await self.engine.stop()
        
        # Flush and remove queues
        logger.info("Flushing and removing queues...")
        while True:
            if connection_manager.flush_and_remove_queues(
                self.settings.exchange_name, 
                self.settings.rabbitmq_username, 
                self.settings.rabbitmq_password
            ):
                break
            time.sleep(1)
        
        # Clear message store
        logger.info("Clearing message store...")
        self.store.clear()
        
        # Restore original state by reloading config
        if self._original_config:
            logger.info("Restoring original state...")
            try:
                await self.engine.load_config(self._original_config)
                logger.info("Original state restored successfully")
            except Exception as e:
                logger.error(f"Failed to restore original state: {e}", exc_info=True)
        
        logger.info("Simulation runner stopped")

    def snapshot(self) -> Dict[str, Any]:
        return self.engine.snapshot()

    async def manual_step(self, ticks: int) -> Dict[str, Any]:
        result = self.engine.step(ticks)
        # Process and send messages even for manual steps
        self._process_sensor_messages(result.get("sensor_messages", {}))
        self._process_sector_states(result.get("sector_states", []))
        self._process_agent_states(result.get("agent_states", []))
        self._process_events(result.get("events", []))
        return result

#    # Processing methods for different message types
    # -----------------------------------------------
    
    def _process_sensor_messages(self, sensor_messages):
        """Process and store sensor messages."""
        for sensor_type_name, payloads in sensor_messages.items():
            try:
                sensor_type = SensorType[sensor_type_name]
                topic = get_topic_for_sensor(sensor_type)
                for payload in payloads:
                    self.store.add_message_to_sent(topic, payload)
            except (KeyError, ValueError) as e:
                logger.warning(f"Unknown sensor type: {sensor_type_name}, error: {e}")

    def _process_sector_states(self, sector_states):
        """Process and store sector states."""
        routing_key = TopicRegistry.SECTOR_STATE.value
        for state in sector_states:
            self.store.add_message_to_sent(routing_key, state)

    def _process_agent_states(self, agent_states):
        """Process and store agent states."""
        topic_map = {
            "forester": TopicRegistry.FORESTER_STATE.value,
            "fire_brigade": TopicRegistry.FIRE_BRIGADE_STATE.value
        }
        
        for agent_state in agent_states:
            agent_type = agent_state.get("type")
            if topic := topic_map.get(agent_type):
                self.store.add_message_to_sent(topic, agent_state)

    def _process_events(self, events):
        """Process and store events."""
        for event in events:
            self.store.add_message_to_sent(TopicRegistry.EVENTS.value, event)
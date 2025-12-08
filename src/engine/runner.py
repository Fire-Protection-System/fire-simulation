import asyncio
import threading
import time
import logging

from typing import Any, Dict
from src.engine.base import SimulationEngine
from src.engine.models.sensors.sensor_type import SensorType
from src.messaging.topics import TopicRegistry, get_topic_for_sensor
from src.rabbitmq.message_store import MessageStore
from src.rabbitmq.pika_client import PikaClient
from src.rabbitmq import producer, consumer, connection_manager

logger = logging.getLogger(__name__)


class EngineRunner:
    def __init__(self, engine: SimulationEngine, settings):
        self.engine = engine
        self.settings = settings
        self.store = MessageStore()
        self.client = PikaClient()
        self._stop = threading.Event()
        self._tick_interval = settings.tick_interval
        self._loop_thread: threading.Thread | None = None
        self._write_threads = []
        self._read_threads = []

    async def start(self, config: Dict[str, Any]) -> None:
        await self.engine.load_config(config)
        await self.engine.start()
        self._setup_queues()

        def loop():
            while not self._stop.is_set():
                try:
                    asyncio.run(self._do_step_and_send())  
                except Exception:
                    pass
                time.sleep(self._tick_interval)

        self._loop_thread = threading.Thread(target=loop, daemon=True)
        self._loop_thread.start()

    def _setup_queues(self):
        while True:
            with self.client.connection_ctx() as (connection, channel):
                if connection and channel:
                    break
            time.sleep(2)

        for topic in TopicRegistry:
            if topic.value.startswith("simulation.control"):
                continue 
            
            producer_kwargs = {
                "exchange":    self.settings.exchange_name,
                "routing_key": topic.value,
                "store":       self.store,
                "username":    self.settings.rabbitmq_username,
                "password":    self.settings.rabbitmq_password
            }
            
            t = threading.Thread(
                target=producer.start_producing_messages, 
                kwargs=producer_kwargs, 
                daemon=True
            )
            
            t.start()
            self._write_threads.append(t)

        for topic in [TopicRegistry.FORESTER_ACTIONS, TopicRegistry.FIRE_BRIGADE_ACTIONS]:
            consumer_kwargs = {
                "queue":    topic.value,
                "store":    self.store,
                "username": self.settings.rabbitmq_username,
                "password": self.settings.rabbitmq_password
            }
            
            t = threading.Thread(
                target=consumer.consume_messages_from_queue, 
                kwargs=consumer_kwargs, 
                daemon=True
            )
            
            t.start()
            self._read_threads.append(t)

    async def _do_step_and_send(self):
        result = await self.engine.step(1)
        
        sensor_messages = result.get("sensor_messages", {})
        for sensor_type_name, payloads in sensor_messages.items():
            try:
                sensor_type = SensorType[sensor_type_name]
                topic = get_topic_for_sensor(sensor_type)
                for payload in payloads:
                    self.store.add_message_to_sent(topic, payload)
            except (KeyError, ValueError) as e:
                logger.warning(f"Unknown sensor type: {sensor_type_name}, error: {e}")

        sector_states = result.get("sector_states", [])
        for state in sector_states:
            self.store.add_message_to_sent(TopicRegistry.SECTOR_STATE.value, state)

        agent_states = result.get("agent_states", [])
        for agent_state in agent_states:
            agent_type = agent_state.get("type")
            if agent_type == "forester":
                self.store.add_message_to_sent(TopicRegistry.FORESTER_STATE.value, agent_state)
            elif agent_type == "fire_brigade":
                self.store.add_message_to_sent(TopicRegistry.FIRE_BRIGADE_STATE.value, agent_state)

        events = result.get("events", [])
        for event in events:
            self.store.add_message_to_sent(TopicRegistry.EVENTS.value, event)

    async def stop(self):
        self._stop.set()
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2)
        
        for t in self._write_threads + self._read_threads:
            if t.is_alive():
                t.join(timeout=1)
        
        await self.engine.stop()
        
        while True:
            if connection_manager.remove_queues(
                self.settings.exchange_name, 
                self.settings.rabbitmq_username, 
                self.settings.rabbitmq_password
            ):
                break
            time.sleep(1)

    def snapshot(self) -> Dict[str, Any]:
        return self.engine.snapshot()

    async def manual_step(self, ticks: int) -> Dict[str, Any]:
        return await self.engine.step(ticks)
import asyncio
import threading
import time

from typing import Any, Dict
from engine.core_engine import CoreSimulationEngine

app_settings = get_settings()

def get_topic_for_sensor(sensor_type: str) -> str:
    mapping = {
        SensorType.TEMPERATURE_AND_AIR_HUMIDITY.name: app_settings.WRITE_QUEUE_TOPICS[2],
        SensorType.WIND_SPEED.name:                   app_settings.WRITE_QUEUE_TOPICS[3],
        SensorType.WIND_DIRECTION.name:               app_settings.WRITE_QUEUE_TOPICS[4],
        SensorType.LITTER_MOISTURE.name:              app_settings.WRITE_QUEUE_TOPICS[5],
        SensorType.PM2_5.name:                        app_settings.WRITE_QUEUE_TOPICS[7],
        SensorType.CO2.name:                          app_settings.WRITE_QUEUE_TOPICS[6],
        SensorType.CAMERA.name:                       app_settings.WRITE_QUEUE_TOPICS[1]
    }
    return mapping.get(sensor_type, app_settings.WRITE_QUEUE_TOPICS[0])

class EngineRunner:
    def __init__(self, settings: SimulationSettings, engine: CoreSimulationEngine):
        self.engine = engine
        self.store = MessageStore()
        self.client = PikaClient()
        self._stop = threading.Event()
        self._tick_interval = settings.TICK_INTERVAL
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

        for q_topic in app_settings.WRITE_QUEUE_TOPICS:

            producer_kargs = {
                "exchange":    app_settings.EXCHANGE_NAME,
                "routing_key": q_topic,
                "store":       self.store,
                "username":    app_settings.RABBITMQ_USERNAME,
                "password":    app_settings.RABBITMQ_PASSWORD
            }

            t = threading.Thread(target=producer.start_producing_messages, kwargs=producer_kargs, daemon=True)
            t.start()
            self._write_threads.append(t)

        for q_topic in app_settings.READ_QUEUE_TOPICS:

            consumer_kargs = {
                "queue":       q_topic,
                "store":       self.store,
                "username":    app_settings.RABBITMQ_USERNAME,
                "password":    app_settings.RABBITMQ_PASSWORD
            }

            t = threading.Thread(target=consumer.consume_messages_from_queue, kwargs=consumer_kargs, daemon=True)
            t.start()
            self._read_threads.append(t)

    async def _do_step_and_send(self):
        result = await self.engine.step(1)
        sensor_messages = result.get("sensor_messages", {})
        sector_states = result.get("sector_states", [])

        for sensor_type, payloads in sensor_messages.items():
            topic = get_topic_for_sensor(sensor_type)
            for p in payloads:
                self.store.add_message_to_sent(topic, p)
                
        for s in sector_states:
            self.store.add_message_to_sent("Sector state topic", s)

    async def stop(self):
        self._stop.set()
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2)
        for t in self._write_threads + self._read_threads:
            if t.is_alive():
                t.join(timeout=1)
        await self.engine.stop()
        while True:
            if connection_manager.remove_queues(EXCHANGE_NAME, RABBITMQ_USERNAME, RABBITMQ_PASSWORD):
                break
            time.sleep(1)
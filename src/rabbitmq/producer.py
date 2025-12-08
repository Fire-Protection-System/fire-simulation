import pika
import logging
import json
import time
import os

from rabbitmq.message_store import MessageStore
from rabbitmq.pika_client import PikaClient
from settings.settings import get_settings


app_settings = get_settings()
logger = logging.getLogger(__name__)
    
def produce_message(
    exchange: str, 
    channel: str, 
    routing_key: str, 
    message: str
) -> None:
    try:
        if channel is None:
            logger.info("Channel is None")
            return

        channel.basic_publish(exchange=exchange, routing_key=routing_key, body=json.dumps(message))
        # if routing_key in ["Forester patrol state topic", "Fire brigades state topic"]:
        #     logger.info(f"Sent message: {message}")

    except Exception as e:
        logger.error(f"Error sending message: {e}")

def start_producing_messages(
    exchange: str, 
    routing_key: str,
    store: MessageStore, 
    username: str, 
    password: str, 
    stop_event
) -> None:
    client = PikaClient()
    try:
        with client.connection_ctx() as (connection, channel):
            if connection is None or channel is None:
                logger.error("Cannot establish connection to RabbitMQ, producer exiting")
                return

            while not stop_event.is_set():
                message = store.get_message_to_sent(routing_key)
                if message:
                    try:
                        channel.basic_publish(exchange=exchange, routing_key=routing_key, body=json.dumps(message))
                    except Exception as e:
                        logger.exception("Error publishing message: %s", e)
                time.sleep(0.5)

    except Exception as e:
        print(f"Connection error: {e}")

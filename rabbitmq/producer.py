import pika
import logging
import json
from simulation.rabbitmq.message_store import MessageStore
import time
import os

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
    credentials = pika.PlainCredentials(username, password)
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host = app_settings.rabbitmq_host,
                port = app_settings.rabbitmq_port,
                credentials = credentials
            )
        )
        channel = connection.channel()
        
        while not stop_event.is_set():
            message = store.get_message_to_sent(routing_key)
            if message:
                produce_message(exchange, channel, routing_key, message)
            time.sleep(0.5)

    except Exception as e:
        print(f"Connection error: {e}")

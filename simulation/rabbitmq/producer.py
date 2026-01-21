import pika
import logging
import json
from simulation.rabbitmq.message_store import MessageStore
import time
import os

logger = logging.getLogger(__name__)
    
RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'rabbitmq-service')
RABBITMQ_PORT = int(os.environ.get('RABBITMQ_PORT', 5672))

def produce_message(exchange, channel, routing_key, message):
    try:
        if channel is None:
            logger.info("Channel is None")
            return
        channel.basic_publish(exchange=exchange, routing_key=routing_key, body=json.dumps(message))
        if routing_key in ["Forester patrol state topic", "Fire brigades state topic"]:
            logger.info(f"Sent message: {message}")
    except Exception as e:
        logger.error(f"Error sending message: {e}")

def start_producing_messages(exchange, routing_key, store: MessageStore, username, password, stop_event):
    credentials = pika.PlainCredentials(username, password)
    while not stop_event.is_set():
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=RABBITMQ_HOST,
                    port=RABBITMQ_PORT,
                    credentials=credentials,
                    connection_attempts=3,
                    retry_delay=5
                )
            )
            channel = connection.channel()
            
            while not stop_event.is_set():
                message = store.get_message_to_sent(routing_key)
                if message:
                    produce_message(exchange, channel, routing_key, message)
                time.sleep(1)
            
            connection.close()
        except Exception as e:
            logger.error(f"Connection error in producer for {routing_key}: {e}")
            if not stop_event.is_set():
                logger.info(f"Retrying connection for {routing_key} in 5 seconds...")
                time.sleep(5)
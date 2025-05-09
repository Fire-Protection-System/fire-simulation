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
        #logger.info(f"Channel: {channel}")
        channel.basic_publish(exchange=exchange, routing_key=routing_key, body=json.dumps(message))
        if routing_key in ["Forester patrol state topic", "Fire brigades state topic"]:
            logger.info(f"Sent message: {message}")
    except Exception as e:
        logger.error(f"Error sending message: {e}")

def start_producing_messages(exchange, routing_key, store: MessageStore, username, password):
    credentials = pika.PlainCredentials(username, password)
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=credentials
            )
        )
        channel = connection.channel()
        
        while True:
            message = store.get_message_to_sent(routing_key)
            if message:
                produce_message(exchange, channel, routing_key, message)
            time.sleep(0.5)
    except Exception as e:
        print(f"Connection error: {e}")
        # Add more robust error handling or logging here
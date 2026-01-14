import pika
import logging
import json
import time
import os

from rabbitmq.message_store import MessageStore
from rabbitmq.pika_client import PikaClient
from settings.communucation_settings import get_communication_settings

'''
    RabbitMQ producer module for sending messages to RabbitMQ exchanges.
'''

RECONNECTED_DELAY   = "${RECONNECTED_DELAY:-2}"
MAX_RECONNECT_DELAY = "${MAX_RECONNECT_DELAY:-30}"

app_settings = get_communication_settings()
logger = logging.getLogger(__name__)
    
def produce_message(
    exchange: str,     
    channel,
    routing_key: str,
    message: dict
) -> bool:
    """
    Publishes a message to the given RabbitMQ exchange and routing key.
    Returns True if successful, False otherwise.
    """
    if channel is None:
        logger.warning("Channel is None, cannot send message.")
        return False
    try:
        channel.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=2)  # Make message persistent
        )
        logger.debug(f"Message sent to {exchange}:{routing_key}")
        return True
    except Exception as e:
        logger.error(f"Error sending message to {exchange}:{routing_key}: {e}")
        return False

def start_producing_messages(
    exchange: str, 
    routing_key: str,
    store: MessageStore, 
    username: str = "", 
    password: str = "", 
    stop_event = None
) -> None:
    '''
        Start producing messages to RabbitMQ exchange with given routing key.
    '''
    logger.info(f"Producer thread starting for routing_key: {routing_key}, exchange: {exchange}")
    client = PikaClient()

    # Load producers configurations
    reconnect_delay     = RECONNECTED_DELAY
    max_reconnect_delay = MAX_RECONNECT_DELAY
    
    while not stop_event.is_set():
        try:
            with client.connection_ctx() as (connection, channel):
                if connection is None or channel is None:
                    logger.error(f"Cannot establish connection to RabbitMQ for {routing_key}, retrying in {reconnect_delay}s")
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                    continue
                
                reconnect_delay = RECONNECTED_DELAY
                logger.info(f"Producer connected for routing_key: {routing_key}")
                
                # Exchange should already be declared by runner._setup_queues()
                # But we declare it here too to ensure it exists (non-passive)
                try:
                    channel.exchange_declare(exchange=exchange, exchange_type='topic', durable=False)
                except Exception as e:
                    logger.warning(f"Exchange {exchange} declaration issue: {e}")

                message_count = 0
                while not stop_event.is_set():
                    try:
                        message = store.get_message_to_sent(routing_key)
                        if message:
                            try:
                                # Message from store is already a dict, serialize it
                                json_body = json.dumps(message) if isinstance(message, dict) else message
                                channel.basic_publish(
                                    exchange=exchange, 
                                    routing_key=routing_key, 
                                    body=json_body,
                                    properties=pika.BasicProperties(delivery_mode=2)
                                )
                                message_count += 1
                                # Only log every 1000 messages to reduce log volume
                                if message_count % 1000 == 0:
                                    logger.debug(f"Published {message_count} messages to {routing_key}")
                            except Exception as e:
                                logger.exception(f"Error publishing message to {routing_key}: {e}")
                                break
                        time.sleep(0.1)
                    except Exception as e:
                        logger.exception(f"Error in producer loop for {routing_key}: {e}")
                        break
        except Exception as e:
            logger.exception(f"Connection error for producer {routing_key}: {e}")
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
    
    logger.info(f"Producer thread stopped for routing_key: {routing_key}")

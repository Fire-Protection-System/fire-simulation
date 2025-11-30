import pika
import logging
import json
from simulation.rabbitmq.message_store import MessageStore
import functools
import os

logger = logging.getLogger(__name__)
app_settings = get_settings()

def callback(
    ch, 
    method, 
    properties, 
    body, 
    store: MessageStore, 
    queue_name: str
) -> None:
    data = json.loads(body.decode('utf-8'))
    store.add_received_message(data, queue_name)
    logger.info(f"Received message {data} from {queue_name}")
    ch.basic_ack(delivery_tag=method.delivery_tag)

def consume_messages_from_queue(queue_name: str, store: MessageStore, username: str, password: str) -> None:
    client = PikaClient()

    try:
        with client.connection_ctx() as (connection, channel):
            if connection is None or channel is None:
                logger.error("Cannot establish connection to RabbitMQ, consumer exiting")
                return

            callback_with_store = functools.partial(callback, store=store, queue_name=queue_name)
            channel.basic_consume(queue=queue_name, on_message_callback=callback_with_store)
            logger.info(f"Waiting for messages in queue: {queue_name}.")
        channel.start_consuming()

    except Exception as e:
        logger.error(f"Error consuming messages: {e}")


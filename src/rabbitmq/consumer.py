import pika
import logging
import json
import functools
import os

from rabbitmq.message_store import MessageStore
from rabbitmq.pika_client import PikaClient
from settings.communucation_settings import get_communication_settings

logger = logging.getLogger(__name__)

app_settings = get_communication_settings()

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
    logger.debug(f"Received message from {queue_name}")
    ch.basic_ack(delivery_tag=method.delivery_tag)

def consume_messages_from_queue(
    queue_name: str,
    store: MessageStore,
    stop_event=None,
    *args,
    **kwargs
) -> None:
    client = PikaClient()

    try:
        with client.connection_ctx() as (connection, channel):
            if connection is None or channel is None:
                logger.error("Cannot establish connection to RabbitMQ, consumer exiting")
                return

            callback_with_store = functools.partial(callback, store=store, queue_name=queue_name)
            channel.basic_consume(queue=queue_name, on_message_callback=callback_with_store)
            logger.debug(f"Waiting for messages in queue: {queue_name}.")
            
            # Check stop event periodically while consuming
            if stop_event:
                import threading
                import time
                def check_stop():
                    while not stop_event.is_set():
                        time.sleep(0.1)
                    # Stop consuming when stop event is set
                    if channel and channel.is_open:
                        try:
                            channel.stop_consuming()
                            logger.info(f"Stopped consuming from queue: {queue_name}")
                        except Exception as e:
                            logger.debug(f"Error stopping consumer: {e}")
                
                stop_checker = threading.Thread(target=check_stop, daemon=True, name=f"StopChecker-{queue_name}")
                stop_checker.start()
            
            # Start consuming - this will block until stop_consuming() is called
            channel.start_consuming()
            logger.info(f"Consumer for queue {queue_name} finished consuming")

    except KeyboardInterrupt:
        logger.info(f"Consumer for queue {queue_name} interrupted")
    except Exception as e:
        if stop_event and stop_event.is_set():
            logger.info(f"Consumer for queue {queue_name} stopped (stop event set)")
        else:
            logger.error(f"Error consuming messages from queue {queue_name}: {e}", exc_info=True)


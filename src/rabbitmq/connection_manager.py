import pika
import logging

from rabbitmq.pika_client import PikaClient
from src.messaging.topics import get_all_topics

logger = logging.getLogger(__name__)

def create_queues(exchange_name: str):
    '''
    Create queues and bind them to topics.
    
    :param exchange_name: str
    '''
    try:
        client = PikaClient()
        conn = client.create_connection()
        channel = conn.channel()

        all_topics = get_all_topics()
        for topic in all_topics:
            queue_name = topic.replace('.', '_')
            channel.queue_declare(queue=queue_name)
            logger.info(f"Queue created: {queue_name}")

        channel.exchange_declare(exchange=exchange_name, exchange_type='topic')

        for topic in all_topics:
            queue_name = topic.replace('.', '_')
            channel.queue_bind(exchange=exchange_name, queue=queue_name, routing_key=topic)
            logger.info(f"Queue '{queue_name}' bound to topic '{topic}'")

        logger.info("All queues and topics are created and bound.")
        return conn, channel

    except Exception as e:
        logger.error(f"Error connecting to RabbitMQ: {e}")
        return None, None

def remove_queues(exchange_name: str, *args, **kwargs):
    '''
    Remove simulation queues and exchange.

    Extra positional/keyword arguments are accepted for backward compatibility
    with older callers that passed username/password, but are ignored.
    '''
    try:
        logger.info("Removing queues and unbinding from topics")
        client = PikaClient()
        conn = client.create_connection()
        channel = conn.channel()
        channel.exchange_delete(exchange=exchange_name)

        all_topics = get_all_topics()
        for topic in all_topics:
            queue_name = topic.replace('.', '_')
            try:
                channel.queue_delete(queue=queue_name)
                logger.info(f"Queue deleted: {queue_name}")
            except Exception as e:
                logger.debug(f"Could not delete queue {queue_name}: {e}")

        try:
            if conn and not conn.is_closed:
                conn.close()
        except Exception:
            pass

        return True

    except Exception as e:
        logger.error(f"Error connecting to RabbitMQ: {e}")
        return False

def flush_and_remove_queues(exchange_name: str, *args, **kwargs):
    '''
    Flush (purge) all messages from queues, then remove queues and exchange.
    This ensures no leftover messages remain when simulation stops.

    Extra positional/keyword arguments are accepted for backward compatibility
    with older callers that passed username/password, but are ignored.
    '''
    try:
        logger.info("Flushing and removing queues")
        client = PikaClient()
        conn = client.create_connection()
        channel = conn.channel()

        all_topics = get_all_topics()
        
        # First, purge all messages from queues
        for topic in all_topics:
            queue_name = topic.replace('.', '_')
            try:
                channel.queue_declare(queue=queue_name, passive=False)
                purged = channel.queue_purge(queue=queue_name)
                logger.info(f"Purged {purged} messages from queue: {queue_name}")
            except Exception as e:
                logger.debug(f"Could not purge queue {queue_name} (may not exist): {e}")

        # Delete exchange
        try:
            channel.exchange_delete(exchange=exchange_name)
            logger.info(f"Exchange deleted: {exchange_name}")
        except Exception as e:
            logger.warning(f"Could not delete exchange {exchange_name}: {e}")

        # Delete queues
        for topic in all_topics:
            queue_name = topic.replace('.', '_')
            try:
                channel.queue_delete(queue=queue_name)
                logger.info(f"Queue deleted: {queue_name}")
            except Exception as e:
                logger.debug(f"Could not delete queue {queue_name} (may not exist): {e}")

        try:
            if conn and not conn.is_closed:
                conn.close()
        except Exception:
            pass

        return True

    except Exception as e:
        logger.error(f"Error flushing and removing queues: {e}")
        return False

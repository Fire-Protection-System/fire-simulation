import pika
import logging

logger = logging.getLogger(__name__)
app_settings = get_settings()

def create_queues(exchange_name, username, password):
    try:
        client = PikaClient()
        with client.connection_ctx() as (connection, channel):
            if connection is None or channel is None:
                logger.error("Cannot establish connection to RabbitMQ, exiting")
                return None, None
                
        for queue_name in app_settings.QUEUE_NAMES:
            channel.queue_declare(queue=queue_name)
            logger.info(f"Queue created: {queue_name}")

        channel.exchange_declare(exchange=exchange_name, exchange_type='topic')

        for topic_name, queue_name in zip(app_settings.TOPIC_NAMES, app_settings.QUEUE_NAMES):
            channel.queue_bind(exchange=exchange_name, queue=queue_name, routing_key=topic_name)
            logger.info(f"Queue '{queue_name}' bound to topic '{topic_name}'")

        logger.info("All queues and topics are created and bound.")

        return connection, channel

    except Exception as e:
        logger.error(f"Error connecting to RabbitMQ: {e}")
        return None, None

def remove_queues(exchange_name, username, password):
    try:
        connection = _connect()
        channel = connection.channel()
        channel.exchange_delete(exchange=exchange_name)

        # Deleting Queues
        for queue_name in app_settings.QUEUE_NAMES:
            channel.queue_delete(queue=queue_name)
            logger.info(f"Queue deleted: {queue_name}")

        return True

    except Exception as e:
        logger.error(f"Error connecting to RabbitMQ: {e}")
        return False

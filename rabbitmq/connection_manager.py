import pika
import logging

logger = logging.getLogger(__name__)
app_settings = get_settings()

def _connection_parameters() -> pika.ConnectionParameters:
    credentials = pika.PlainCredentials(
        app_settings.RABBITMQ_USERNAME, 
        app_settings.RABBITMQ_PASSWORD
    )
   
    return pika.ConnectionParameters(
        host=app_settings.RABBITMQ_HOST,
        port=app_settings.RABBITMQ_PORT,
        credentials=credentials,
        heartbeat=60,
        blocked_connection_timeout=30
    )

def _connect() -> pika.BlockingConnection:
    return pika.BlockingConnection(_connection_parameters())

def create_queues(exchange_name, username, password):
    try:
        connection = _connect()
        channel = connection.channel()

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

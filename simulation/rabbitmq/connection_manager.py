import pika
import logging

logger = logging.getLogger(__name__)

RABBITMQ_HOST = 'rabbitmq-service' 
RABBITMQ_PORT = 5672
RABBITMQ_USERNAME = 'guest'
RABBITMQ_PASSWORD = 'guest'

QUEUE_NAMES = [
    "Forester patrol action queue",
    "Forester patrol state queue",
    "Camera queue",
    "Temp and air humidity queue",
    "Wind speed queue",
    "Wind direction queue",
    "Litter moisture queue",
    "CO2 queue",
    "PM2.5 queue",
    "Fire brigades action queue",
    "Fire brigades state queue"
]

TOPIC_NAMES = [
    "Forester patrol action topic",
    "Forester patrol state topic",
    "Camera topic",
    "Temp and air humidity topic",
    "Wind speed topic",
    "Wind direction topic",
    "Litter moisture topic",
    "CO2 topic",
    "PM2.5 topic",
    "Fire brigades action topic",
    "Fire brigades state topic"
]

def create_queues(exchange_name, username, password):
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD)

        connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials
        ))
        channel = connection.channel()

        # Creating Queues
        for queue_name in QUEUE_NAMES:
            channel.queue_declare(queue=queue_name)
            logger.info(f"Queue created: {queue_name}")

        # Creating Exchange (Topic Type)
        channel.exchange_declare(exchange=exchange_name, exchange_type='topic')

        # Binding Queues to Topics
        for topic_name, queue_name in zip(TOPIC_NAMES, QUEUE_NAMES):
            channel.queue_bind(exchange=exchange_name, queue=queue_name, routing_key=topic_name)
            logger.info(f"Queue '{queue_name}' bound to topic '{topic_name}'")

        logger.info("All queues and topics are created and bound.")

        return connection, channel

    except Exception as e:
        logger.error(f"Error connecting to RabbitMQ: {e}")
        return None, None

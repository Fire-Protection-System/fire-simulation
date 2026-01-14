import os

# Centralised settings for RabbitMQ connection and topics
# Can be configured via environment variables

from messaging.topics import ControlTopics, SimulationTopics
from pydantic import Field
from pydantic_settings import BaseSettings
from .rabbitmq_queues import QUEUE_NAMES, TOPIC_NAMES

DEFAULT_RABBITMQ_HOST = "127.0.0.1"
DEFAULT_RABBITMQ_PORT = 5672
DEFAULT_RABBITMQ_USER = "guest"
DEFAULT_RABBITMQ_PASS = "guest"
DEFAULT_EXCHANGE      = "fire-simulation-exchange"

class CommunicationSettings(BaseSettings):
    """RabbitMQ connection settings"""
    rabbitmq_host: str           = Field(DEFAULT_RABBITMQ_HOST, env="RABBITMQ_HOST")
    rabbitmq_port: int           = Field(DEFAULT_RABBITMQ_PORT, env="RABBITMQ_PORT")
    rabbitmq_username: str       = Field(DEFAULT_RABBITMQ_USER, env="RABBITMQ_USERNAME")
    rabbitmq_password: str       = Field(DEFAULT_RABBITMQ_PASS, env="RABBITMQ_PASSWORD")
    exchange_name: str           = Field(DEFAULT_EXCHANGE, env="FIRE_SIMULATION_EXCHANGE_NAME")
    published_topics: list[str]  = Field(default_factory=list)
    subscribed_topics: list[str] = Field(default_factory=list)

class SimulatorCommunicationSettings(CommunicationSettings):
    """Settings specific to simulator component"""
    published_topics: list[str] = Field(
        default_factory=lambda: [t.value for t in SimulationTopics]
    )
    subscribed_topics: list[str] = Field(
        default_factory=lambda: [t.value for t in ControlTopics]
    )

def get_communication_settings() -> CommunicationSettings:
    return CommunicationSettings()

def get_simulator_settings() -> SimulatorCommunicationSettings:
    return SimulatorCommunicationSettings()
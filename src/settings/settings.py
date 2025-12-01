import os

# Centralised settings for RabbitMQ connection and topics
# Can be configured via environment variables

from pydantic import BaseSettings, Field, validator

""" Load settings from environment variables or use defaults """

DEFAULT_RABBITMQ_HOST = "rabbitmq-service"
DEFAULT_RABBITMQ_PORT = 5672
DEFAULT_RABBITMQ_USER = "guest"
DEFAULT_RABBITMQ_PASS = "guest"
DEFAULT_EXCHANGE      = "fire_updates"
DEFAULT_TICK_INTERVAL = 5.0

DEFAULT_WRITE_QUEUE_TOPICS = [
    "Forester patrol state topic",
    "Camera topic",
    "Temp and air humidity topic",
    "Wind speed topic",
    "Wind direction topic",
    "Litter moisture topic",
    "CO2 topic",
    "PM2.5 topic",
    "Fire brigades state topic",
    "Recommended action topic",
    "Sector state topic"
]

DEFAULT_READ_QUEUE_TOPICS = [
    "Forester patrol action queue",
    "Fire brigades action queue"
]

from dataclasses import dataclass
from typing import List
import os, json

class Settings(BaseSettings):
    rabbitmq_host:     str = Field(DEFAULT_RABBITMQ_HOST, env="RABBITMQ_HOST")
    rabbitmq_port:     int = Field(DEFAULT_RABBITMQ_PORT, env="RABBITMQ_PORT")
    rabbitmq_username: str = Field(DEFAULT_RABBITMQ_USER, env="RABBITMQ_USERNAME")
    rabbitmq_password: str = Field(DEFAULT_RABBITMQ_PASS, env="RABBITMQ_PASSWORD")
    exchange_name:     str = Field(DEFAULT_EXCHANGE, env="FIRE_SIMULATION_EXCHANGE_NAME")

    write_queue_topics: List[str] = Field(
        default_factory=lambda: DEFAULT_WRITE_QUEUE_TOPICS,
        env="WRITE_QUEUE_TOPICS",
    )

    read_queue_topics: List[str] = Field(
        default_factory=lambda: DEFAULT_READ_QUEUE_TOPICS,
        env="READ_QUEUE_TOPICS",
    )

    tick_interval: float = Field(DEFAULT_TICK_INTERVAL, env="TICK_INTERVAL")


def get_settings() -> Settings:
    return Settings()

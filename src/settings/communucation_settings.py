import os

# Centralised settings for RabbitMQ connection and topics
# Can be configured via environment variables

from pydantic import Field
from pydantic_settings import BaseSettings

DEFAULT_RABBITMQ_HOST = "rabbitmq-service"
DEFAULT_RABBITMQ_PORT = 5672
DEFAULT_RABBITMQ_USER = "guest"
DEFAULT_RABBITMQ_PASS = "guest"
DEFAULT_EXCHANGE      = "fire-simulation-exchange"


class Settings(BaseSettings):
    rabbitmq_host:     str   = Field(DEFAULT_RABBITMQ_HOST, env="RABBITMQ_HOST")
    rabbitmq_port:     int   = Field(DEFAULT_RABBITMQ_PORT, env="RABBITMQ_PORT")
    rabbitmq_username: str   = Field(DEFAULT_RABBITMQ_USER, env="RABBITMQ_USERNAME")
    rabbitmq_password: str   = Field(DEFAULT_RABBITMQ_PASS, env="RABBITMQ_PASSWORD")
    exchange_name:     str   = Field(DEFAULT_EXCHANGE,      env="FIRE_SIMULATION_EXCHANGE_NAME")


def get_communication_settings() -> Settings:
    return Settings()

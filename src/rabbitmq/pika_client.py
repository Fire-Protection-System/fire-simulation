import pika
import logging
from contextlib import contextmanager
from typing import Iterator, Tuple, Optional

from settings.settings import get_settings

logger = logging.getLogger(__name__)

class PikaClient:
    def __init__(self):
        self._settings = get_settings()

    def connection_parameters(self) -> pika.ConnectionParameters:
        creds = pika.PlainCredentials(
            self._settings.rabbitmq_username, 
            self._settings.rabbitmq_password
        )
        
        return pika.ConnectionParameters(
            host=self._settings.rabbitmq_host,
            port=self._settings.rabbitmq_port,
            credentials=creds,
            heartbeat=60,
            blocked_connection_timeout=30,
        )

    def create_connection(self) -> pika.BlockingConnection:
        params = self.connection_parameters()
        return pika.BlockingConnection(params)

    def create_channel(self) -> pika.channel.Channel:
        conn = self.create_connection()
        return conn.channel()

    @contextmanager
    def connection_ctx(self) -> Iterator[Tuple[Optional[pika.BlockingConnection], Optional[pika.channel.Channel]]]:
        conn = None
        ch = None
        try:
            conn = self.create_connection()
            ch = conn.channel()
            yield conn, ch
        except Exception as e:
            logger.exception("Pika connection error: %s", e)
            yield None, None
        finally:
            try:
                if conn and not conn.is_closed:
                    conn.close()
            except Exception:
                pass
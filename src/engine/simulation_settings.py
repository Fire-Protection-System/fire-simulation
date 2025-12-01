from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SimulationSettings:
    READ_QUEUE_TOPICS: List[str]
    WRITE_QUEUE_TOPICS: List[str]
    EXCHANGE_NAME: str
    RABBITMQ_USERNAME: str
    RABBITMQ_PASSWORD: str
    RABBITMQ_HOST: Optional[str] = None
    RABBITMQ_PORT: Optional[int] = None
    TICK_INTERVAL: float = 5.0
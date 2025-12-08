from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SimulationSettings:
    TICK_INTERVAL: float = 5.0
    FIRE_FIGHTERS_MULTIPLIER = 5
    FIRE_LEVEL_MULTIPLIER = 1

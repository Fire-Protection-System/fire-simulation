import os
from pydantic import Field
from pydantic_settings import BaseSettings

TICK_INTERVAL               = 0.1    # base sim tick (kept small for smooth movement)
FIRE_FIGHTERS_MULTIPLIER    = 5      # used in extinguishing logic (see fire_brigade)
FIRE_LEVEL_MULTIPLIER       = 1      # reserved for future tuning of fire growth
FIRE_SPREAD_PROB_MULTIPLIER = 0.05   # was 0.1 – slower spreading between sectors
WAIT_FOR_SUPPORT            = False
SUPPORT_TIMEOUT             = 10.0
SECTOR_UPDATE_INTERVAL      = 10     # was 5 – sector fire updates 2x rzadziej
AGENT_UPDATES_PER_SIM_TICK  = 3      # było 2 – agenci aktualizują się częściej niż ogień

class SimulationSettings(BaseSettings):
    tick_interval: float               = Field(TICK_INTERVAL,               env="TICK_INTERVAL")
    fire_fighters_multiplier: int      = Field(FIRE_FIGHTERS_MULTIPLIER,    env="FIRE_FIGHTERS_MULTIPLIER")
    fire_level_multiplier: int         = Field(FIRE_LEVEL_MULTIPLIER,       env="FIRE_LEVEL_MULTIPLIER")
    fire_spread_prob_multiplier: float = Field(FIRE_SPREAD_PROB_MULTIPLIER, env="FIRE_SPREAD_PROB_MULTIPLIER")
    wait_for_support: bool             = Field(WAIT_FOR_SUPPORT,            env="WAIT_FOR_SUPPORT")
    support_timeout: float             = Field(SUPPORT_TIMEOUT,             env="SUPPORT_TIMEOUT")
    sector_update_interval: int        = Field(SECTOR_UPDATE_INTERVAL,      env="SECTOR_UPDATE_INTERVAL")
    agent_updates_per_sim_tick: int    = Field(AGENT_UPDATES_PER_SIM_TICK,  env="AGENT_UPDATES_PER_SIM_TICK")

def get_simulation_settings() -> SimulationSettings:
    return SimulationSettings()

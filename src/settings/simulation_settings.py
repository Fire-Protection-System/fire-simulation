import os

'''
    Centralized setting for the simulation parameters. 
    These settings can be adjusted via environment variables.
    For now our idea is to have all the setting related to simulation here, 
    disregarding engine models/engine components used. 
'''

from pydantic import Field
from pydantic_settings import BaseSettings

'''
    SIMPLE SIMULATION ENGINE SETTINGS
    ----------------------------------------
    TICK_INTERVAL            : time interval (in seconds) for each simulation tick
    FIRE_FIGHTERS_MULTIPLIER : multiplier for fire fighters effectiveness in the simulation in same sector. 
                               The higher the multiplier the faster the fire fighters will extinguish the fire.
    FIRE_LEVEL_MULTIPLIER    : multiplier for the fire level increase in the simulation
'''

TICK_INTERVAL            = 1.0           
FIRE_FIGHTERS_MULTIPLIER = 5             
FIRE_LEVEL_MULTIPLIER    = 1             
WAIT_FOR_SUPPORT         = False
SUPPORT_TIMEOUT          = 10.0

     
class SimulationSettings(BaseSettings):
    tick_interval: float          = Field(TICK_INTERVAL,            env="TICK_INTERVAL")
    fire_fighters_multiplier: int = Field(FIRE_FIGHTERS_MULTIPLIER, env="FIRE_FIGHTERS_MULTIPLIER")
    fire_level_multiplier: int    = Field(FIRE_LEVEL_MULTIPLIER,    env="FIRE_LEVEL_MULTIPLIER")
    wait_for_support: bool        = Field(WAIT_FOR_SUPPORT,         env="WAIT_FOR_SUPPORT")
    support_timeout: float        = Field(SUPPORT_TIMEOUT,          env="SUPPORT_TIMEOUT")

def get_simulation_settings() -> SimulationSettings:
    return SimulationSettings()
    
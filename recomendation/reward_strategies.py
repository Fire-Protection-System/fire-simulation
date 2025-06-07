from dataclasses import dataclass
from typing import List
from enum import Enum

@dataclass
class RewardConfigBaseline:
    LOST_SECTOR_PENALTY = 50
    EXTINGUISHED_FIRES_REWARD = 100
    SPREAD_PREVENTION_REWARD = 20
    BURNED_SECTOR_PENALTY = 20
    name = "baseline"

@dataclass
class RewardConfigBalanced:
    LOST_SECTOR_PENALTY = 35
    EXTINGUISHED_FIRES_REWARD = 250
    SPREAD_PREVENTION_REWARD = 60
    BURNED_SECTOR_PENALTY = 18
    name = "balanced"

@dataclass
class RewardConfigAggressive:
    LOST_SECTOR_PENALTY = 30
    EXTINGUISHED_FIRES_REWARD = 350
    SPREAD_PREVENTION_REWARD = 80
    BURNED_SECTOR_PENALTY = 15
    name = "aggressive"

@dataclass
class RewardConfigPreventive:
    LOST_SECTOR_PENALTY = 40
    EXTINGUISHED_FIRES_REWARD = 200
    SPREAD_PREVENTION_REWARD = 100
    BURNED_SECTOR_PENALTY = 20
    name = "preventive"

@dataclass
class RewardConfigConservative:
    LOST_SECTOR_PENALTY = 25
    EXTINGUISHED_FIRES_REWARD = 180
    SPREAD_PREVENTION_REWARD = 45
    BURNED_SECTOR_PENALTY = 12
    name = "conservative"

@dataclass
class RewardConfigExtreme:
    LOST_SECTOR_PENALTY = 20
    EXTINGUISHED_FIRES_REWARD = 500
    SPREAD_PREVENTION_REWARD = 120
    BURNED_SECTOR_PENALTY = 10
    name = "extreme"

REWARD_CONFIGS = {
    "baseline": RewardConfigBaseline(),
    "balanced": RewardConfigBalanced(),
    "aggressive": RewardConfigAggressive(),
    "preventive": RewardConfigPreventive(),
    "conservative": RewardConfigConservative(),
    "extreme": RewardConfigExtreme()
}

def get_config_by_name(name: str):
    if name in REWARD_CONFIGS:
        return REWARD_CONFIGS[name]
    else:
        return RewardConfigBaseline()

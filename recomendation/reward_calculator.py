from dataclasses import dataclass
from typing import List
from enum import Enum

from simulation.sectors.fire_state import FireState

@dataclass
class RewardConfig:
    LOST_SECTOR_PENALTY = 50
    EXTINGUISHED_FIRES_REWARD = 20
    SPREAD_PREVENTION_REWARD = 100

class RewardCalculator:
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
    
    def calculate_reward(self, state) -> float:
        """
        Calculate total reward based on current state.
        """
        penalties = self._calculate_penalties(state)
        rewards = self._calculate_rewards(state)
        total_reward = penalties + rewards
        
        return total_reward
    
    def _calculate_penalties(self, state) -> float:
        lost_sectors_penalty = self._calculate_lost_sectors_penalty(state)
        active_fire_penalty = self._calculate_active_fire_penalty(state)
        
        return lost_sectors_penalty + active_fire_penalty
    
    def _calculate_rewards(self, state) -> float:
        extinguished_fires_reward = self._calculate_extinguished_fires_reward(state)
        spread_prevention_reward = self._calculate_spread_prevention_reward(state)
        
        return extinguished_fires_reward + spread_prevention_reward
    
    def _calculate_lost_sectors_penalty(self, state) -> float:
        lost_sectors_count = sum(
            1 for sector in state.sectors 
            if sector.fire_state == FireState.LOST
        )
        
        penalty = -self.config.LOST_SECTOR_PENALTY * lost_sectors_count
        return penalty
    
    def _calculate_active_fire_penalty(self, state) -> float:
        total_fire_intensity = sum(
            sector.fire_level 
            for sector in state.sectors 
            if sector.fire_state == FireState.ACTIVE
        )
        
        penalty = -total_fire_intensity
        return penalty
    
    def _calculate_extinguished_fires_reward(self, state) -> float:
        successfully_extinguished_count = sum(
            1 for sector in state.sectors
            if self._is_successfully_extinguished(sector)
        )
        
        reward = self.config.EXTINGUISHED_FIRES_REWARD * successfully_extinguished_count
        return reward
    
    def _calculate_spread_prevention_reward(self, state) -> float:
        firebreak_sectors_count = sum(
            1 for sector in state.sectors
            if self._is_effective_firebreak(sector, state)
        )
        
        reward = self.config.SPREAD_PREVENTION_REWARD * firebreak_sectors_count
        return reward
    
    def _is_successfully_extinguished(self, sector) -> bool:
        return (sector.fire_state == FireState.INACTIVE and 
                sector.burn_level > 0)
    
    def _is_effective_firebreak(self, sector, state) -> bool:
        if sector.fire_state != FireState.INACTIVE:
            return False
        
        adjacent_sectors = state.map.get_adjacent_sectors(sector)
        has_burning_neighbor = any(
            neighbor.fire_state == FireState.ACTIVE
            for neighbor, *_ in adjacent_sectors
        )
        
        return has_burning_neighbor
    
    def get_reward_breakdown(self, state) -> dict:

        penalties = self._calculate_penalties(state)
        rewards = self._calculate_rewards(state)
        
        return {
            'lost_sectors_penalty': self._calculate_lost_sectors_penalty(state),
            'active_fire_penalty': self._calculate_active_fire_penalty(state),
            'extinguished_fires_reward': self._calculate_extinguished_fires_reward(state),
            'spread_prevention_reward': self._calculate_spread_prevention_reward(state),
            'total_reward': penalties + rewards
        }
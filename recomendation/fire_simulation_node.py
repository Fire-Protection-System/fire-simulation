import copy
import random
from itertools import product, permutations
from typing import List, Tuple, Optional, Set, Dict, FrozenSet, Any, NamedTuple
import numpy as np
from collections import defaultdict, namedtuple
import collections

import heapq
import collections
import numpy as np

from recomendation.node import Node
from simulation.sectors.sector import Sector
from simulation.sectors.fire_state import FireState
from simulation.fire_spread.coef_generator import calculate_beta
from simulation.fire_spread.wind import Wind
from simulation.forest_map import ForestMap
from simulation.sectors.geographic_direction import GeographicDirection
from simulation.agent_state import AGENT_STATE
from simulation.agent import Agent
from simulation.fire_brigades.fire_brigade import FireBrigade
from simulation.forester_patrols.forester_patrol import ForesterPatrol

class Position(NamedTuple):
    latitude: float
    longitude: float

class FireSimulationNode(Node):
    __slots__ = ('sectors', 'sector_states', 'agents', 'agent_states', 'time_step', 
                'max_steps', 'action', 'max_brigades', 'map', 'wind', '_children_cache',
                '_sector_lookup', '_active_sector_ids')
    
    def __init__(
        self,
        sectors: List[Sector],
        agents: List[Agent],
        time_step: int,
        max_steps: int = 10,
        action: Optional[List[Tuple[int, int]]] = None,
        max_brigades: int = 2,
        forest_map: Optional[ForestMap] = None,
        wind: Optional[Wind] = None,

        sector_states=None,
        agent_states=None,
        _sector_lookup=None,
        _active_sector_ids=None
    ):
        self.sectors = sectors
        
        """Caching every state/object possible to optimize computations."""
        if sector_states is None:
            self.sector_states = frozenset((
                s.sector_id,
                s.fire_state, 
                s.fire_level, 
                s.burn_level, 
                s._number_of_fire_brigades
            ) for s in sectors)
        else:
            self.sector_states = sector_states
                    
        if agent_states is None:
            self.agent_states = tuple((
                a.state, 
                a.location.latitude, 
                a.location.longitude,
                getattr(a.destination, 'latitude', None),
                getattr(a.destination, 'longitude', None)
            ) for a in agents)
        else:
            self.agent_states = agent_states
        
        if _sector_lookup is None:
            self._sector_lookup = {
                s.sector_id: s for s in sectors
            }
        else:
            self._sector_lookup = _sector_lookup
            
        if _active_sector_ids is None:
            self._active_sector_ids = frozenset(s.sector_id for s in sectors if s.fire_state == FireState.ACTIVE)
        else:
            self._active_sector_ids = _active_sector_ids
        
        self.time_step = time_step
        self.max_steps = max_steps
        self.action = tuple(action) if action else None
        self.max_brigades = max_brigades
        self.map = forest_map
        self.wind = wind or Wind()
        self._children_cache = None
        self.agents = agents
        
    def __hash__(self):
        return hash((self.time_step, self.sector_states, self.agent_states))

    def __eq__(self, other):
        return (isinstance(other, FireSimulationNode) and 
                self.time_step == other.time_step and
                self.sector_states == other.sector_states and
                self.agent_states == other.agent_states)
    
    def _calculate_distance(self, loc1, loc2):
        """Calculate distance between two locations using numpy for speed"""
        lat_diff = loc1.latitude - loc2.latitude
        lon_diff = loc1.longitude - loc2.longitude
        return np.sqrt(lat_diff**2 + lon_diff**2)

    def find_children(self):
        """Find Best Children using optimized heuristics and cached values"""
        if self._children_cache is not None:
            return self._children_cache

        if not self._active_sector_ids:
            self._children_cache = set()
            return self._children_cache

        active_sectors = [self._sector_lookup[sid] for sid in self._active_sector_ids]
        num_agents = len(self.agents)
        children = set()

        sector_threat = {
            s.sector_id: s.fire_level * (1 + s.burn_level)
            for s in active_sectors
        }
        required_brigades = {
            s.sector_id: s.required_fire_brigades()
            for s in active_sectors
        }

        agent_locations = {
            i: (self.agents[i].location.latitude, self.agents[i].location.longitude)
            for i in range(num_agents)
        }
        sector_locations = {
            s.sector_id: self.map.get_sector_location(s)
            for s in active_sectors
        }

        agent_to_sector_distances = {
            (i, sid): np.hypot(agent_locations[i][0] - sector_locations[sid].latitude,
                            agent_locations[i][1] - sector_locations[sid].longitude)
            for i in range(num_agents)
            for sid in sector_threat
        }

        top_k = max(1, min(len(active_sectors), num_agents * 2, int(len(self.sectors) * 0.1)))
        top_sectors = heapq.nlargest(top_k, active_sectors, key=lambda s: sector_threat[s.sector_id])

        agent_scores = []
        for i in range(num_agents):
            score = sum(
                sector_threat[s.sector_id] / max(0.1, agent_to_sector_distances[(i, s.sector_id)])
                for s in top_sectors
            )
            agent_scores.append((i, score))
        agent_scores.sort(key=lambda x: x[1], reverse=True)

        assigned_sectors = set()
        threat_actions = []
        sector_brigade_counts = collections.defaultdict(int)

        for i, _ in agent_scores:
            best_options = [
                (sid, sector_threat[sid] / (1 + agent_to_sector_distances[(i, sid)] * 0.5))
                for sid in sector_threat
                if sid not in assigned_sectors
            ]
            if best_options:
                best_sid, _ = max(best_options, key=lambda x: x[1])
                if sector_brigade_counts[best_sid] < required_brigades[best_sid]:
                    threat_actions.append((i, best_sid))
                    assigned_sectors.add(best_sid)
                    sector_brigade_counts[best_sid] += 1

        if threat_actions:
            children.add(self._apply_step(threat_actions))

        if num_agents > 1:
            for s in top_sectors[:min(3, len(top_sectors))]:
                sid = s.sector_id
                required = required_brigades[sid]
                if required == 0:
                    continue

                closest_agents = sorted(
                    [(i, agent_to_sector_distances[(i, sid)]) for i in range(num_agents)],
                    key=lambda x: x[1]
                )[:min(required, num_agents)]

                actions = [(i, sid) for i, _ in closest_agents]
                if actions:
                    children.add(self._apply_step(actions))

        seen_assignments = set()
        max_random = min(10, 30 - len(children))
        sector_ids = list(sector_threat.keys())
        agent_sector_probs = {}

        for i in range(num_agents):
            probs = np.array([
                sector_threat[sid] / (1 + agent_to_sector_distances[(i, sid)])
                for sid in sector_ids
            ])
            if probs.sum() > 0:
                probs = probs / probs.sum()
                agent_sector_probs[i] = (sector_ids, probs)

        for _ in range(max_random):
            actions = []
            temp_brigade_counts = collections.defaultdict(int)
            for i in range(num_agents):
                if i in agent_sector_probs:
                    sids, probs = agent_sector_probs[i]
                    chosen = np.random.choice(sids, p=probs)
                    if temp_brigade_counts[chosen] < required_brigades[chosen]:
                        actions.append((i, chosen))
                        temp_brigade_counts[chosen] += 1

            actions_sorted = tuple(sorted(actions))
            if actions and actions_sorted not in seen_assignments:
                seen_assignments.add(actions_sorted)
                children.add(self._apply_step(actions))

        self._children_cache = children
        return children

    def find_random_child(self):
        """Find a random child using cached active sectors"""
        if not self._active_sector_ids:
            return None
            
        active_sectors = [self._sector_lookup[sid] for sid in self._active_sector_ids]
        num_agents = len(self.agents)
        actions = []
        
        if len(active_sectors) >= num_agents:
            chosen_sectors = random.sample(active_sectors, num_agents)
            for i, sector in enumerate(chosen_sectors):
                actions.append((i, sector.sector_id))
        else:
            for i in range(num_agents):
                actions.append((i, random.choice(active_sectors).sector_id))
        
        return self._apply_step(actions)

    def _apply_step(self, actions: List[Tuple[int, int]]):
        """Apply actions and create new state with optimized object creation"""
        sectors_to_modify = set()
        for _, sector_id in actions:
            sectors_to_modify.add(sector_id)
            
        new_sectors_map = {}
        new_sectors = []

        for s in self.sectors:
            cloned = s.clone()
            new_sectors_map[s.sector_id] = cloned
            new_sectors.append(cloned)
        
        new_map = self.map.clone()
        new_map.update_sectors(new_sectors)
        
        new_agents = []
        for i, agent in enumerate(self.agents):
            agent_actions = [a for a in actions if a[0] == i] 
            if agent_actions:
                new_agent = agent.clone()
                new_agents.append(new_agent)
            else:
                new_agents.append(agent)
        
        for agent_idx, sector_id in actions:
            agent = new_agents[agent_idx]
            dest_sector = new_map.get_sector(sector_id)
            dest_position = new_map.get_sector_location(dest_sector)
            agent.set_state_travelling(dest_position)
        
        for agent in new_agents:
            if agent.state == AGENT_STATE.TRAVELLING:
                if self._update_position(agent):
                    agent._location = agent.destination
                
                    if agent.destination == agent.base_location:
                        agent.set_state_available()
                    else:
                        sec = new_map.find_sector(agent.location)
                        sec._number_of_fire_brigades += 1
                        agent.set_state_executing()

        updated_sector_ids = set()
        for agent in new_agents:
            if agent.state == AGENT_STATE.EXECUTING:
                sec = new_map.find_sector(agent.location)

                if sec.sector_id not in updated_sector_ids:
                    if sec is not new_sectors_map[sec.sector_id]:
                        sec = sec.clone()
                        new_sectors_map[sec.sector_id] = sec
                    updated_sector_ids.add(sec.sector_id)

                if isinstance(agent, FireBrigade):
                    sec.update_extinguish_level()
                    
                    if hasattr(sec, "fire_intensity") and sec.extinguish_level >= sec.fire_intensity:
                        agent.set_state_available()
                
                elif hasattr(agent, "is_task_finished") and agent.is_task_finished(sec):
                    agent.set_state_available()

        new_active_sector_ids = set()
        for sector in new_sectors:
            sector.update_sector()
            if sector.fire_state == FireState.ACTIVE:
                new_active_sector_ids.add(sector.sector_id)
        
        self._spread_fire(new_sectors, new_map, copy.copy(self.wind), new_active_sector_ids)

        new_sector_states = frozenset((s.sector_id, s.fire_state, s.fire_level, s.burn_level, 
                                      s._number_of_fire_brigades) for s in new_sectors)
                                      
        new_agent_states = tuple((a.state, a.location.latitude, a.location.longitude,
                                getattr(a.destination, 'latitude', None),
                                getattr(a.destination, 'longitude', None)) for a in new_agents)
                                
        new_sector_lookup = {s.sector_id: s for s in new_sectors}
        new_active_sector_ids = frozenset(s.sector_id for s in new_sectors if s.fire_state == FireState.ACTIVE)

        return FireSimulationNode(
            sectors=new_sectors,
            agents=new_agents,
            time_step=self.time_step + 1,
            max_steps=self.max_steps,
            action=actions,
            max_brigades=self.max_brigades,
            forest_map=new_map,
            wind=copy.copy(self.wind),
            
            sector_states=new_sector_states,
            agent_states=new_agent_states,
            _sector_lookup=new_sector_lookup,
            _active_sector_ids=new_active_sector_ids
        )

    def calculate_reward(self, state):
        """Calculate reward strategy (optional)"""

        LOST_SECTOR_PENALTY = 20
        EXTINGUISHED_FIRES_REWARD = 20
        SPREAD_PREVENTION_REWARD = 20
        
        lost_penalty = -LOST_SECTOR_PENALTY * sum(1 for s in state.sectors if s.fire_state == FireState.LOST)
        
        fire_penalty = -sum(s.fire_level for s in state.sectors if s.fire_state == FireState.ACTIVE)
        
        extinguished_reward = EXTINGUISHED_FIRES_REWARD * sum(1 for s in state.sectors 
                                    if s.fire_state == FireState.INACTIVE and s.burn_level > 0)

        spread_prevention = SPREAD_PREVENTION_REWARD * sum(1 for s in state.sectors 
                                if s.fire_state == FireState.INACTIVE and
                                any(n.fire_state == FireState.ACTIVE 
                                    for n, _ in state.map.get_adjacent_sectors(s)))
        
        return lost_penalty + fire_penalty + extinguished_reward + spread_prevention

    def is_terminal(self):
        """Optimized terminal state check using cached active sectors"""
        return self.time_step >= self.max_steps or not self._active_sector_ids

    def reward(self):
        """Calculate reward"""
        return self.calculate_reward(self)

    def _update_position(self, agent: Agent) -> bool:
        """Update agent position with optimized movement calculation"""
        delta = 0.005

        dest_lat, dest_lon = agent.destination.latitude, agent.destination.longitude
        curr_lat, curr_lon = agent.location.latitude, agent.location.longitude
        
        # Calculate direction once
        lat_diff = dest_lat - curr_lat
        lon_diff = dest_lon - curr_lon
        
        # Use min/max for bounds checking
        agent._location.latitude += min(delta, lat_diff) if lat_diff > 0 else max(-delta, lat_diff)
        agent._location.longitude += min(delta, lon_diff) if lon_diff > 0 else max(-delta, lon_diff)

        # Check for arrival
        return (abs(lat_diff) <= 0.001 and abs(lon_diff) <= 0.001)

    def _spread_fire(self, sectors: List[Sector], forest_map: ForestMap, wind: Wind, active_sector_ids=None):
        """Optimized fire spread simulation using cached data structures"""

        # Use cached sectors if possible
        if active_sector_ids is None:
            active_sectors = [s for s in sectors if s.fire_state == FireState.ACTIVE]
        else:
            active_sectors = [s for s in sectors if s.sector_id in active_sector_ids]
        
        # Lookup for storing sectors to activate
        sectors_to_activate = []
        sector_lookup = {s.sector_id: s for s in sectors}
        
        # Pre-compute all adjacency sectors
        adjacency_map = {}
        for sector in active_sectors:
            adjacency_map[sector.sector_id] = forest_map.get_adjacent_sectors(sector)
        
        # Process each currently burning sector
        # Calculate probability with simulation function
        for sector in active_sectors:
            for neighbor, direction in adjacency_map[sector.sector_id]:
                if neighbor.fire_state == FireState.INACTIVE:
                    prob = calculate_beta(wind, neighbor.sector_type, direction)
                    if random.random() < prob:
                        sectors_to_activate.append(neighbor)
        
        # Apply fires to chosen sectors
        for sector in sectors_to_activate:
            sector.start_fire()
            
        wind.update_wind()

    def simulate_steps_without_action(self, steps: int) -> List[Sector]:
        """Simulate future steps without agent actions - optimized version"""

        active_sectors = {s.sector_id for s in self.sectors if s.fire_state == FireState.ACTIVE}
        
        clones_map = {}
        clones = []
        
        sectors_to_clone = set(active_sectors)
        for s in self.sectors:
            if s.fire_state == FireState.ACTIVE:
                for neighbor, _ in self.map.get_adjacent_sectors(s):
                    sectors_to_clone.add(neighbor.sector_id)
        
        for s in self.sectors:
            if s.sector_id in sectors_to_clone:
                clone = s.clone()
                clones_map[s.sector_id] = clone
                clones.append(clone)
            else:
                clones_map[s.sector_id] = s
                clones.append(s)
        
        fmap = self.map.clone()
        fmap.update_sectors(clones)
        w = copy.copy(self.wind)
        
        active_sector_ids = frozenset(s.sector_id for s in clones if s.fire_state == FireState.ACTIVE)

        for _ in range(steps):
            sectors_to_update = set()
            
            for s_id in active_sector_ids:
                s = clones_map[s_id]
                if s is self._sector_lookup.get(s_id):
                    s = s.clone()
                    clones_map[s_id] = s
                    idx = clones.index(self._sector_lookup[s_id])
                    clones[idx] = s
                s.update_sector()
                
                for neighbor, _ in fmap.get_adjacent_sectors(s):
                    sectors_to_update.add(neighbor.sector_id)
                
            self._spread_fire(clones, fmap, w, active_sector_ids)
            
            active_sector_ids = frozenset(s.sector_id for s in clones if s.fire_state == FireState.ACTIVE)

        return clones

    def simulate_steps_with_action(self, steps: int, actions: List[Tuple[int, int]]) -> Tuple[List[Sector], List[Agent]]:
        """Simulate future steps with specified agent actions - optimized version"""
        affected_sectors = set()
        for _, sector_id in actions:
            affected_sectors.add(sector_id)
        
        active_sectors = {s.sector_id for s in self.sectors if s.fire_state == FireState.ACTIVE}
        affected_sectors.update(active_sectors)
        
        for s in self.sectors:
            if s.fire_state == FireState.ACTIVE:
                for neighbor, _ in self.map.get_adjacent_sectors(s):
                    affected_sectors.add(neighbor.sector_id)
        
        clones_map = {}
        clones = []
        for s in self.sectors:
            if s.sector_id in affected_sectors:
                clone = s.clone()
                clones_map[s.sector_id] = clone
                clones.append(clone)
            else:
                clones_map[s.sector_id] = s
                clones.append(s)
        
        fmap = self.map.clone()
        fmap.update_sectors(clones)
        w = copy.copy(self.wind)
        
        ags = [i for i in self.agents]
        # for i, agent in enumerate(self.agents):
        #     if any(a[0] == i for a in actions):
        #         ags.append(agent.clone())
        #     else:
        #         ags.append(agent)
        
        for idx, sec_id in actions:
            ag = ags[idx]
            dest = fmap.get_sector_location(fmap.get_sector(sec_id))
            ag.set_state_travelling(dest)
        
        active_sector_ids = frozenset(s.sector_id for s in clones if s.fire_state == FireState.ACTIVE)
        
        for _ in range(steps):
            updated_sectors = set()
            for ag in ags:
                if ag.state == AGENT_STATE.TRAVELLING:
                    if self._update_position(ag):
                        ag._location = ag.destination
                        
                        if ag.destination == ag.base_location:
                            ag.set_state_available()
                        else:
                            ag.set_state_executing()
                            if isinstance(ag, FireBrigade):
                                sec = fmap.find_sector(ag.location)
                                sec._number_of_fire_brigades += 1
                    
                if ag.state == AGENT_STATE.EXECUTING:
                    sector = fmap.find_sector(ag.location)
                    if sector and sector.sector_id not in updated_sectors:
                        if sector is self._sector_lookup.get(sector.sector_id):
                            sector = sector.clone()
                            try:
                                idx = clones.index(self._sector_lookup[sector.sector_id])
                                clones[idx] = sector
                                clones_map[sector.sector_id] = sector
                            except ValueError:
                                clones.append(sector)
                                clones_map[sector.sector_id] = sector
                        updated_sectors.add(sector.sector_id)
            
            sectors_to_update = active_sector_ids.union(updated_sectors)
            
            for s_id in sectors_to_update:
                if s_id in clones_map:
                    s = clones_map[s_id]
                    if s is self._sector_lookup.get(s_id):
                        s = s.clone()
                        try:
                            idx = clones.index(self._sector_lookup[s_id])
                            clones[idx] = s
                        except ValueError:
                            clones.append(s)
                        clones_map[s_id] = s
                    s.update_sector()
            
            self._spread_fire(clones, fmap, w, active_sector_ids)
            active_sector_ids = frozenset(s.sector_id for s in clones if s.fire_state == FireState.ACTIVE)
        
        return clones, ags
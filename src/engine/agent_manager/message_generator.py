from datetime import datetime

from engine.models.agents.agent import Agent
from engine.models.agents.fire_brigade import FireBrigade
from engine.models.agents.fire_brigade_state import FIREBRIGADE_STATE
from engine.models.agents.forest_patrols_state import FORESTERPATROL_STATE
from engine.models.agents.forester_patrol import ForesterPatrol
from engine.models.map.sector import Sector


def generate_traveling_message(agent: Agent):
    if isinstance(agent, FireBrigade):

        return {
                "fireBrigadeId": agent.fire_brigade_id,
                "state": FIREBRIGADE_STATE.TRAVELLING.name,
                "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                "location": {
                    "longitude": agent.location.longitude,
                    "latitude": agent.location.latitude
                }
            }
    
    elif isinstance(agent, ForesterPatrol):
        return  {
                "foresterPatrolId": agent.forester_patrol_id,
                "state": FORESTERPATROL_STATE.TRAVELLING.name,
                "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                "location": {
                    "longitude": agent.location.longitude,
                    "latitude": agent.location.latitude
                }
            }
    
def generate_message_available(agent : Agent):
    if isinstance(agent, FireBrigade):
        return {
            "fireBrigadeId": agent.fire_brigade_id,
            "state": FIREBRIGADE_STATE.AVAILABLE.name,
            "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "location": {
                "longitude": agent.location.longitude,
                "latitude": agent.location.latitude
            }
        }
    
    elif isinstance(agent, ForesterPatrol):

        return {
            "foresterPatrolId" : agent.forester_patrol_id,
            "state" : FORESTERPATROL_STATE.AVAILABLE.name,
            "timestamp" : datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "location": {
                "longitude" : agent.location.longitude,
                "latitude" : agent.location.latitude
            }
        }
    
def generate_message_extinguished(agent : Agent, sector: Sector):
    if isinstance(agent, FireBrigade):
        return  {
            "fireBrigadeId": agent.fire_brigade_id,
            "state": FIREBRIGADE_STATE.AVAILABLE.name,
            "fireState": sector.fire_state.name,
            "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "location": {
                "longitude" : agent.location.longitude,
                "latitude" : agent.location.latitude
            }
        }
    
def generate_message_extinguishing(agent : Agent):
    if isinstance(agent, FireBrigade):
        return  {
            "fireBrigadeId": agent.fire_brigade_id,
            "state": FIREBRIGADE_STATE.EXTINGUISHING.name,
            "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "location": {
                "longitude" : agent.location.longitude,
                "latitude" : agent.location.latitude
            }
        }

def generate_message_patrolling(agent : Agent, sector : Sector):
    if isinstance(agent, ForesterPatrol):
        return {
            "foresterPatrolId" : agent.forester_patrol_id,
            "state" : FORESTERPATROL_STATE.PATROLLING.name,
            "timestamp" : datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            "location": {
                "longitude" : agent.location.longitude,
                "latitude" : agent.location.latitude
            },
            "sectorState": sector.fire_state.name
        }
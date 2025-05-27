import logging
import random
from typing import List, Tuple
from queue import Queue

import threading
import time
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from simulation.forest_map import ForestMap
from threading import Thread, Event
from simulation.sectors.sector import Sector
from simulation.sectors.fire_state import FireState
from simulation.fire_spread.coef_generator import calculate_beta
from simulation.fire_spread.wind import Wind
from simulation.sectors.geographic_direction import GeographicDirection
from simulation.agent_manager.agent_manager import AgentManager

from simulation.rabbitmq import producer, consumer, connection_manager
from simulation.rabbitmq.message_store import MessageStore
from simulation.sensors.sensor_type import SensorType
from simulation.fire_brigades.fire_brigade_state import FIREBRIGADE_STATE

from recomendation.mcts_test import predict

logger = logging.getLogger(__name__)

EXCHANGE_NAME = "fire_updates"
USERNAME = "guest"
PASSWORD = "guest"

WRITE_QUEUE_TOPICS = [
    "Forester patrol state topic",
    "Camera topic",
    "Temp and air humidity topic",
    "Wind speed topic",
    "Wind direction topic",
    "Litter moisture topic",
    "CO2 topic",
    "PM2.5 topic",
    "Fire brigades state topic",
    "Recommended action topic"
]

READ_QUEUE_TOPICS = [
    "Forester patrol action queue",
    "Fire brigades action queue"
]

stop_event = threading.Event()

def stop_simulation():
    stop_event.set()

def get_topic_for_sensor(sensor_type: str) -> str:
    # Indeksy odpowiadające typom sensorów w SensorType
    topic_mapping = {
        SensorType.TEMPERATURE_AND_AIR_HUMIDITY.name: WRITE_QUEUE_TOPICS[2],
        SensorType.WIND_SPEED.name: WRITE_QUEUE_TOPICS[3],
        SensorType.WIND_DIRECTION.name: WRITE_QUEUE_TOPICS[4],
        SensorType.LITTER_MOISTURE.name: WRITE_QUEUE_TOPICS[5],
        SensorType.PM2_5.name: WRITE_QUEUE_TOPICS[7],
        SensorType.CO2.name: WRITE_QUEUE_TOPICS[6],
        SensorType.CAMERA.name: WRITE_QUEUE_TOPICS[1]
    }
    return topic_mapping.get(sensor_type, "Unknown topic")

def run_simulation(configuration):
    store = MessageStore()
    read_threads = []
    write_threads = []

    # MCTS predictions results
    prediction_queue = Queue()

    def prediction_worker():
        while not stop_event.is_set():
            try:
                forest_map_clone = map.clone()
                recommended_actions = predict(forest_map_clone)
                available_agents = [a.fire_brigade_id for a in forest_map_clone.fireBrigades if a.state == FIREBRIGADE_STATE.AVAILABLE]

                if recommended_actions:
                    action_queue = "Recommended action topic"
                    action_message = {
                        "timestamp": time.time(),
                        "recommendedActions": [
                            { "unitId": int(unit_id), "sectorId": int(sector_id) }
                            for unit_id, sector_id in recommended_actions if unit_id in available_agents
                        ],
                        "priority": "high"
                    }
                    store.add_message_to_sent(action_queue, action_message)
                    logger.info(f"MCTS recommended actions: {recommended_actions}")

            except Exception as e:
                logger.error(f"Error in MCTS prediction: {str(e)}")

            time.sleep(5)

    #===================Get connection and channel===================

    while(1):
        connection, channel = connection_manager.create_queues(EXCHANGE_NAME, USERNAME, PASSWORD)
        logger.info("Queues have been created!")
        if(connection and channel):
            break
        logger.error("Error while connecting to RabbitMQ. Trying to reconnect.")
        time.sleep(5)

    #===================Threads with producing and consuming===================

    for index, queue in enumerate(WRITE_QUEUE_TOPICS):
        write_threads.append(Thread(target=producer.start_producing_messages, args=(EXCHANGE_NAME, queue, store, USERNAME, PASSWORD, stop_event)))
        write_threads[index].start()
        logger.info(f"Producer for {queue} has started working.")
    
    for index, queue in enumerate(READ_QUEUE_TOPICS):
        read_threads.append(Thread(target=consumer.consume_messages_from_queue, args=( queue, store, USERNAME, PASSWORD)))
        read_threads[index].start()
        logger.info(f"Consumer for {queue} has started working.")
    
    #===================Get configuration===================

    map = ForestMap.from_conf(configuration)
    agents_manager = AgentManager(map, store)

    orderProcessingThread = Thread(target=agents_manager.start_processing_orders)
    orderProcessingThread.start()

    # Start the prediction thread
    prediction_thread = Thread(target=prediction_worker)
    prediction_thread.daemon = True
    prediction_thread.start()
    logger.info("MCTS prediction thread started")

    #===================SIMULATION===================
    wind = Wind()

    all_sectors: List[Sector] = [item for sublist in map.sectors for item in sublist]
    sectors_on_fire: List[Sector] = []
    sectors_on_fire.append(map.start_new_fire())

    while not stop_event.is_set():

        for sector in sectors_on_fire:
            sector.update_sector()

            neighbours: List[Tuple[Sector, GeographicDirection]]
            neighbours = map.get_adjacent_sectors(sector) #all neighbours

            new_sectors_on_fire: List[Sector] = []
            for neighbour in neighbours:
                if(neighbour[0].fire_state is FireState.INACTIVE):
                    probability = calculate_beta(wind, neighbour[0].sector_type, neighbour[1]) 
                    if random.random() < probability:
                        neighbour[0].start_fire()
                        new_sectors_on_fire.append(neighbour[0])

        sectors_on_fire = list(filter(lambda sector: sector.fire_state is FireState.ACTIVE, sectors_on_fire))
        sectors_on_fire.extend(new_sectors_on_fire)        

        wind.update_wind()

        for sector in all_sectors:
            sector.update_sensors()
            
            for sensor_type, jsons in sector.make_jsons().items():
                queue = get_topic_for_sensor(sensor_type)
                for json in jsons:
                    store.add_message_to_sent(queue, json)
        
        agents_manager.update_agents_states()
        time.sleep(5)


    # ===================Stop Threads with producing and consuming===================

    for thread in write_threads:
        thread.join()
    prediction_thread.join()

    #===================Remove queues===================

    while(1):
        result = connection_manager.remove_queues(EXCHANGE_NAME, USERNAME, PASSWORD)
        logger.info("Queues have been removed!")
        if(result):
            break
        logger.error("Error while connecting to RabbitMQ. Trying to reconnect.")
        time.sleep(5)
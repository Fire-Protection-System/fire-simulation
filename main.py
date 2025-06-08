from flask import Flask, request
import simulation_engine
import threading

import logging
from logging_config import setup_logging

import sys

app = Flask(__name__)

def background_task(data):
    simulation_engine.run_simulation(data)

@app.route('/run_simulation', methods=['POST'])
def run():
    data = request.get_json()
    print("Received data:", data)

    simulation_engine.stop_event.clear()
    thread = threading.Thread(target=background_task, args=(data,))
    thread.start()
        
    return 'Simulation is running'


@app.route('/stop_simulation', methods=['POST'])
def stop():
    simulation_engine.stop_simulation()

    return 'Simulation stopped'

if __name__ == '__main__':
    setup_logging("fire-simulation")

    # Logger Configuration
    """
    TODO: 
        Change logging logic to silence logs from recommendation's module
        since they use same shared models, i.e. Sectors.
    """
    simulation_logger = logging.getLogger("simulation")
    simulation_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    simulation_logger.addHandler(handler)
    logging.getLogger("recommendation").setLevel(logging.CRITICAL + 1)

    app.run(debug=True, host='0.0.0.0', port=5000)
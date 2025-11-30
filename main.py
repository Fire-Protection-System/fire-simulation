import asyncio
import logging
import sys
import threading
from flask import Flask, request, jsonify

from engine.simple_simulation_engine import SimpleSimulationEngine
from engine.runner import EngineRunner
from logger.logging_config import setup_logging
from settings.config import get_settings

app = Flask(__name__)

app_settings = get_settings()

sim_settings = SimulationSettings(
    READ_QUEUE_TOPICS = app_settings.read_queue_topics,
    WRITE_QUEUE_TOPICS = app_settings.write_queue_topics,
    EXCHANGE_NAME = app_settings.exchange_name,
    RABBITMQ_USERNAME = app_settings.rabbitmq_username,
    RABBITMQ_PASSWORD = app_settings.rabbitmq_password,
    RABBITMQ_HOST = app_settings.rabbitmq_host,
    RABBITMQ_PORT = app_settings.rabbitmq_port,
    TICK_INTERVAL = app_settings.tick_interval,
)

engine = SimpleSimulationEngine()
runner = EngineRunner(engine, settings = sim_settings)

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None

def _get_event_loop() -> asyncio.AbstractEventLoop:
    global _loop, _loop_thread
    if _loop is None or not _loop.is_running():
        _loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(target=_loop.run_forever, daemon=True)
        _loop_thread.start()
    return _loop


def _run_async(coro):
    loop = _get_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)


@app.route('/run_simulation', methods=['POST'])
def run():
    data = request.get_json()
    print("Received data:", data)

    if runner.engine.is_running():
        return jsonify({"status": "error", "message": "Simulation already running"}), 400

    try:
        _run_async(runner.start(data))
        return jsonify({"status": "ok", "message": "Simulation started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/stop_simulation', methods=['POST'])
def stop():
    """Zatrzymaj symulację."""
    if not runner.engine.is_running():
        return jsonify({"status": "error", "message": "Simulation not running"}), 400

    try:
        _run_async(runner.stop())
        return jsonify({"status": "ok", "message": "Simulation stopped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/step', methods=['POST'])
def step():
    """Wykonaj ręcznie n kroków symulacji (domyślnie 1)."""
    data = request.get_json() or {}
    ticks = data.get("ticks", 1)

    if not runner.engine.is_running():
        return jsonify({"status": "error", "message": "Simulation not running"}), 400

    try:
        result = _run_async(runner.manual_step(ticks))
        return jsonify({
            "status": "ok",
            "ticks": ticks,
            "sectors_on_fire": len(result.get("sector_states", [])),
            "sensor_messages_count": sum(len(v) for v in result.get("sensor_messages", {}).values())
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/snapshot', methods=['GET'])
def snapshot():
    """Pobierz aktualny stan symulacji."""
    try:
        snap = runner.snapshot()
        return jsonify({"status": "ok", "snapshot": snap})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({
        "status": "ok",
        "running": runner.engine.is_running()
    })


if __name__ == '__main__':
    setup_logging("fire-simulation")

    simulation_logger = logging.getLogger("simulation")
    simulation_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    simulation_logger.addHandler(handler)

    logging.getLogger("recommendation").setLevel(logging.CRITICAL + 1)

    print("Starting Fire Simulation API...")
    print("Endpoints:")
    print("  POST /run_simulation  - start simulation with config")
    print("  POST /stop_simulation - stop simulation")
    print("  POST /step            - manual step (body: {ticks: n})")
    print("  GET  /snapshot        - get current state")
    print("  GET  /health          - health check")

    app.run(debug=True, host='0.0.0.0', port=5000)
import json
import logging
import threading
import time
from datetime import datetime
from typing import Optional

from src.rabbitmq.message_store import message_store
from src.messaging.topics import TopicRegistry

logger = logging.getLogger(__name__)


class SupportWorker:
    """Support worker suitable for import into core package (src.engine.support).

    Mirrors `fire-support/support_worker.py` but lives inside `src` for easier imports and testing.
    """

    def __init__(self, algorithm: str = "mcts", poll_interval: float = 0.2):
        self._algorithm = algorithm
        self._poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("SupportWorker started (src)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self):
        while self._running:
            try:
                msg = message_store.get_received_message(TopicRegistry.SUPPORT_ANALYSIS_REQUESTS.value)
                if msg:
                    logger.debug("SupportWorker (src) received analysis request")
                    try:
                        parsed = json.loads(msg)
                    except Exception:
                        logger.exception("Failed to parse analysis request")
                        continue
                    result = self.process_message(parsed)
                    if result:
                        self._publish_recommendation(result)
            except Exception:
                logger.exception("SupportWorker loop error")
            time.sleep(self._poll_interval)

    def process_message(self, message: dict) -> Optional[dict]:
        sim_id = message.get("simulationId", "unknown")
        payload = message.get("payload", {})

        telemetry = payload.get("telemetry", [])
        recommendations = []
        for entry in telemetry:
            if entry.get("fire", False):
                recommendations.append({
                    "agentId": entry.get("nearestAgentId", "unknown"),
                    "type": "move_to",
                    "location": entry.get("location"),
                    "confidence": 0.5
                })

        if not recommendations:
            return None

        rec = {
            "schemaVersion": "1.0",
            "timestamp": datetime.now().isoformat(),
            "simulationId": sim_id,
            "recommendations": recommendations,
            "meta": {"generatedBy": "SupportWorker(src)", "algorithm": self._algorithm}
        }
        return rec

    def _publish_recommendation(self, recommendation: dict):
        try:
            message_store.add_message_to_sent(TopicRegistry.SUPPORT_RECOMMENDATIONS.value, json.dumps(recommendation))
            logger.info("Published recommendation to SUPPORT_RECOMMENDATIONS (src)")
        except Exception:
            logger.exception("Failed to publish recommendation")

import logging
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MessageStore:
    def __init__(self):
        self.messages_to_sent = defaultdict(deque)
        self.received_messages = defaultdict(deque) 
        self.lock = threading.Lock() 

    def add_received_message(self, message: str,  queue_name: str) -> None:
        with self.lock:
            self.received_messages[queue_name].append(message)
            logger.debug(f"Received message: {message}")

    def add_message_to_sent(self, queue_name: str, message: str) -> None:
        """Dodaje wiadomość do określonej kolejki."""
        with self.lock:
            self.messages_to_sent[queue_name].append(message)

    def get_message_to_sent(self, queue_name: str) -> str:
        """Pobiera i usuwa najstarszą wiadomość z danej kolejki."""
        with self.lock:
            if self.messages_to_sent[queue_name]:
                oldest_message = self.messages_to_sent[queue_name].popleft()
                return oldest_message
            else:
                return None

    def get_sent_message(self):
        pass

    def get_received_message(self, queue_name):
        with self.lock:
            if self.received_messages[queue_name]:
                oldest_message = self.received_messages[queue_name].popleft()  # Pobiera i usuwa najstarszą wiadomość
                logger.debug(f"Retrieved oldest received message: {oldest_message} from  queue: {queue_name}")
                return oldest_message
            else:
                return None

    def clear(self):
        """Clear all messages from the store."""
        with self.lock:
            self.messages_to_sent.clear()
            self.received_messages.clear()
            logger.info("Message store cleared")

message_store = MessageStore()


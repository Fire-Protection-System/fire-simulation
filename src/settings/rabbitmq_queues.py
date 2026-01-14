"""
DEPRECATED: Use src.messaging.topics.TopicRegistry instead.

This module is kept for backward compatibility only.
All new code should use TopicRegistry from src.messaging.topics.

Migration guide:
    Old: from src.settings.rabbitmq_queues import TOPIC_NAMES
    New: from src.messaging.topics import TopicRegistry, get_all_topics

    Old: topic = TOPIC_NAMES[0]
    New: topic = TopicRegistry.FIRE_BRIGADE_STATE.value
"""

import warnings
from src.messaging.topics import TopicRegistry, get_all_topics

# Emit deprecation warning
warnings.warn(
    "rabbitmq_queues.py is deprecated. Use src.messaging.topics.TopicRegistry instead.",
    DeprecationWarning,
    stacklevel=2
)

# Backward compatibility aliases
QUEUE_NAMES = get_all_topics()
TOPIC_NAMES = QUEUE_NAMES  # Alias for legacy code

# provider/chat_group.py
import threading
import uuid

class ChatGroupManager:
    """Simple thread‑safe manager for a single active chat group ID."""
    def __init__(self):
        self._lock = threading.Lock()
        self._group_id = None

    def get(self):
        with self._lock:
            if self._group_id is None:
                # Generate a random ID similar to browser examples
                self._group_id = str(uuid.uuid4()).replace("-", "")[:24]
            return self._group_id

    def set(self, new_id: str):
        with self._lock:
            if new_id and new_id != self._group_id:
                self._group_id = new_id

# Global instance – can be shared across all requests for simplicity.
# For multi‑user deployments, use request‑scoped storage.
default_manager = ChatGroupManager()

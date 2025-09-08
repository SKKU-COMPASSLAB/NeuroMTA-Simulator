__all__ = [
    "Lock",
    "TicketLock",
]


class Lock:
    def __init__(self):
        self._key = None
    
    def acquire(self, key: str):
        if self._key is not None:
            raise Exception(f"[ERROR] Cannot acquire lock for key '{key}' as it is already acquired by '{self._key}'")
        
        self._key = key
        
    def release(self, key: str):
        if not self.is_locked_with(key):
            raise Exception(f"[ERROR] Cannot release lock for key '{key}' as it is not currently locked")
        
        self._key = None
        
    def is_locked(self) -> bool:
        return self._key is not None
    
    def is_locked_with(self, key: str) -> bool:
        if self._key is None:
            return False
        return self._key == key


class TicketLock:
    def __init__(self, ticket_max: int = 2 ** 32 - 1):
        self._now_serving = 0
        self._next_ticket = 0
        self._ticket_max  = ticket_max
        
        self._key_ticket_mappings: dict[str, int] = {}

    def acquire(self, key: str):
        if key in self._key_ticket_mappings:
            raise Exception(f"[ERROR] Cannot acquire lock for key '{key}' as it is already acquired")
        
        t = self._next_ticket
        self._increase_next_ticket()
        self._key_ticket_mappings[key] = t
        
    def release(self, key: str):
        if not self.is_locked_with(key):
            raise Exception(f"[ERROR] Cannot release lock for key '{key}' as it is not currently locked")
        
        self._key_ticket_mappings.pop(key, None)
        self._increase_now_serving()
        
    def is_acquired_with(self, key: str) -> bool:
        return key in self._key_ticket_mappings

    def is_locked_with(self, key: str) -> bool:
        if key not in self._key_ticket_mappings:
            return False
        return self._now_serving == self._key_ticket_mappings.get(key)

    def _increase_now_serving(self):
        self._now_serving += 1

        if self._now_serving > self._ticket_max:
            self._now_serving = 0
            
    def _increase_next_ticket(self):
        self._next_ticket += 1

        if self._next_ticket > self._ticket_max:
            self._next_ticket = 0
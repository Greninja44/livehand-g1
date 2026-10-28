"""Latest-value buffer: writers overwrite, readers never queue stale items."""
import threading
import time


class LatestBox:
    def __init__(self):
        self._cond = threading.Condition()
        self._item = None
        self._seq = 0

    def put(self, item):
        with self._cond:
            self._item = item
            self._seq += 1
            self._cond.notify_all()

    def get(self):
        """Return (seq, item) without blocking; (0, None) if nothing was ever put."""
        with self._cond:
            return self._seq, self._item

    def wait_newer(self, last_seq, timeout=0.1):
        """Block until an item newer than last_seq exists (or timeout). Returns (seq, item) or (last_seq, None)."""
        end = time.monotonic() + timeout
        with self._cond:
            while self._seq <= last_seq:
                rem = end - time.monotonic()
                if rem <= 0:
                    return last_seq, None
                self._cond.wait(rem)
            return self._seq, self._item

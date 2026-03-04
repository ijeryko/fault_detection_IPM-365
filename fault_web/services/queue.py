import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class EmailJob:
    fn: Callable[[], None]

class EmailQueue:
    def __init__(self, enabled: bool, logger, maxsize: int = 1000):
        self.enabled = enabled
        self.logger = logger
        self.q: "queue.Queue[EmailJob]" = queue.Queue(maxsize=maxsize)
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if not self.enabled:
            self.logger.info("EmailQueue disabled (mailer not configured).")
            return
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.logger.info("EmailQueue started.")

    def enqueue(self, job: EmailJob):
        if not self.enabled:
            return
        try:
            self.q.put_nowait(job)
        except queue.Full:
            self.logger.warning("EmailQueue full; dropping job.")

    def _run(self):
        while True:
            job = self.q.get()
            try:
                job.fn()
            except Exception as e:
                self.logger.exception(f"Email job failed: {e}")
            finally:
                self.q.task_done()
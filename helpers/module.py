import time
from collections.abc import Callable
from threading import Thread, Event
from queue import Queue


class Module:
    def __init__(self,
            queue_in: Queue | None,
            target: Callable[[list], list],
            queue_out: Queue | None,
            batch_size: int = 1,
            timeout: float = 0.1,
        ) -> None:
        """Architecture module that get an object from queue_in, execute target, and push the returned object into queue_out"""

        assert batch_size >= 0
        assert timeout > 0

        # Initialize object variables
        self.target = target
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.batch_size = batch_size
        self.timeout = timeout
        self._thread = Thread(target=self.loop, daemon=True)
        self._run_event = Event() # Marks when the thread have to execute
        self._end_event = Event() # Marks when the module have ended operations
        self._exception: BaseException | None = None

        # Start thread
        self._thread.start()


    def start(self) -> None:
        """Reset initial state of the flags and start thread"""
        self._last_operation = False # During target execution, indicates that the last elements is been passed to target
        self._exception = None
        self._end_event.clear()
        self._run_event.set()


    def queue_in_end(self) -> None:
        """Have to be called after the last element is pushed into the queue_in"""
        if self.queue_in is not None:
            self.queue_in.put(END_QUEUE)
        else:
            self._last_operation = True


    @property
    def last_operation(self) -> bool:
        """Indicates if the last element is been retrieved from the queue_in"""
        return self._last_operation
    

    def wait(self) -> None:
        """Wait for the end of operations"""
        self._end_event.wait()
        if self._exception is not None:
            raise self._exception


    def loop(self) -> None:
        """Thread function"""

        # Initialize cache
        cache = []

        # Execute when the module is started
        self._run_event.wait()
        while True:
            try:

                # If there is the queue_in
                if self.queue_in is not None:

                    # If cache is not enough, wait for other objects from queue
                    while (len(cache) < self.batch_size) and (not self.last_operation):
                        cache.append(self.queue_in.get())
                        while not self.queue_in.empty():
                            cache.append(self.queue_in.get())

                        # Check for end queue
                        if cache[-1] == END_QUEUE:
                            cache = cache[:-1]
                            self._last_operation = True
                
                # If not queue_in, timeout
                else:
                    time.sleep(self.timeout)

                # Execute target function
                results = self.target(cache[:self.batch_size])

                # Push results into queue_out if there is queue_out
                if self.queue_out is not None:
                    for res in results:
                        self.queue_out.put(res)

                # Update cache
                cache = cache[self.batch_size:]

                # If finished and cache empty, stop execution
                if (self.last_operation) and (not cache):
                    self._run_event.clear()
                    self._end_event.set()
                    self._run_event.wait()

            except BaseException as err:
                self._exception = err
                self._run_event.clear()
                self._end_event.set()
                self._run_event.wait()


END_QUEUE = object() # Queue sentinel

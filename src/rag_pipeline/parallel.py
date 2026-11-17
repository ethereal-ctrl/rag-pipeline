"""Bounded thread scheduling for independent remote workloads."""

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice


def completed_work[T, U](items: Iterable[T], work: Callable[[T], U], workers: int) -> Iterator[U]:
    """Yield in completion order, keeping at most one submitted job per worker."""
    if workers == 1:
        yield from map(work, items)
        return
    remaining = iter(items)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        pending = {pool.submit(work, item) for item in islice(remaining, workers)}
        while pending:
            ready, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                yield future.result()
            pending.update(pool.submit(work, item) for item in islice(remaining, len(ready)))
    finally:
        pool.shutdown(cancel_futures=True)

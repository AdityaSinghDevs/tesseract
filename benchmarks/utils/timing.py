import time
from contextlib import contextmanager
from typing import Iterator, Optional

def now()-> float:
    """
    Return a high-resolution, monotonic timestamp suitable for benchmarking.
    """
    return time.perf_counter()

@contextmanager
def measure_time()-> Iterator[float]:
    """
    Context manager to measure elapsed wall-clock time for a code block.
    Usage:
        with measure_time() as elapsed:
            run_some_code()
        print(elapsed)
    """

    start = now()
    yield lambda:now() - start


def measure_duration(fn, *args, **kwargs)-> float:
    """
    Measure execution time of a callable.

    Args:
        fn: Callable to execute
        *args, **kwargs: Arguments passed to the callable

    Returns:
        Elapsed time in seconds.
    """
    start = now()
    fn(*args, **kwargs)
    end = now()
    return end - start

def measure_peak_gpu_memory()-> Optional[int]:
    """
    Return peak GPU memory allocated (in bytes) during the last operation.

    Returns None if CUDA is not available.
    """
    try: 
        import torch
    except ImportError:
        return None
    
    if not torch.cuda.is_available():
        return None
    
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()
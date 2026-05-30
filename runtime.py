import os
import resource
import time


MEMORY_LIMIT_GB = float(os.getenv("WATCHOUT_MEMORY_LIMIT_GB", "6"))
MEMORY_LIMIT_BYTES = int(MEMORY_LIMIT_GB * 1024 * 1024 * 1024)
NATIVE_THREADS = os.getenv("WATCHOUT_NATIVE_THREADS", "1")


def configure_native_runtime() -> None:
    os.environ["OMP_NUM_THREADS"] = NATIVE_THREADS
    os.environ["OMP_THREAD_LIMIT"] = NATIVE_THREADS
    os.environ["MKL_NUM_THREADS"] = NATIVE_THREADS
    os.environ["OPENBLAS_NUM_THREADS"] = NATIVE_THREADS
    os.environ["BLIS_NUM_THREADS"] = NATIVE_THREADS
    os.environ["VECLIB_MAXIMUM_THREADS"] = NATIVE_THREADS
    os.environ["NUMEXPR_NUM_THREADS"] = NATIVE_THREADS
    os.environ["MALLOC_ARENA_MAX"] = "1"
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")


def apply_memory_limit() -> None:
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = MEMORY_LIMIT_BYTES
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    print(f"[{time.strftime('%H:%M:%S')}] 메모리 제한: {limit / 1024 / 1024 / 1024:.2f}GB")

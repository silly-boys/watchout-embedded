import os
import resource
import time


MEMORY_LIMIT_GB = float(os.getenv("WATCHOUT_MEMORY_LIMIT_GB", "6"))
MEMORY_LIMIT_BYTES = int(MEMORY_LIMIT_GB * 1024 * 1024 * 1024)


def configure_native_runtime() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")


def apply_memory_limit() -> None:
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = MEMORY_LIMIT_BYTES
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    print(f"[{time.strftime('%H:%M:%S')}] 메모리 제한: {limit / 1024 / 1024 / 1024:.2f}GB")

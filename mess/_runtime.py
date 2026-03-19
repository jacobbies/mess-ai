"""Runtime guards for native-library interoperability."""

from __future__ import annotations

import os
import sys
from typing import Any


def configure_macos_openmp_runtime() -> None:
    """Stabilize mixed Torch+FAISS workflows on macOS.

    Local research flows in this repo routinely import both Torch and FAISS in the
    same process. On macOS arm64, those libraries can initialize duplicate OpenMP
    runtimes and abort unless the process opts into the duplicate-runtime fallback
    and conservative thread counts. Callers can still override these defaults via
    environment variables before import.
    """
    if sys.platform != "darwin":
        return

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def configure_faiss_runtime(faiss_module: Any) -> Any:
    """Apply FAISS thread caps that match the OpenMP guard defaults."""
    if sys.platform != "darwin":
        return faiss_module

    omp_threads_raw = os.environ.get("OMP_NUM_THREADS", "1")
    try:
        omp_threads = max(1, int(omp_threads_raw))
    except ValueError:
        omp_threads = 1

    if hasattr(faiss_module, "omp_set_num_threads"):
        faiss_module.omp_set_num_threads(omp_threads)

    return faiss_module

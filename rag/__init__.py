"""Voice-enabled RAG over ai4bharat/MSMARCO-XI.

This module exists mostly to make the process survive its own dependencies.

torch, faiss and scikit-learn each ship their own copy of ``libomp``, and on
macOS/arm64 that produces two distinct failures, both silent until they are not:

1. Importing faiss before torch makes the first CPU forward pass segfault. The
   import order is therefore pinned here rather than left to whichever module
   happens to load first.
2. faiss's first parallel region aborts the process with ``OMP: Error #15``
   when a second OpenMP runtime is already initialised.

``scripts/fix_macos_openmp.py`` resolves the second by symlinking the duplicate
runtimes to torch's, which is the real fix; Linux images share the system
OpenMP and need nothing. faiss is additionally pinned to one thread because a
single-query HNSW search is serial regardless, and a web server should not have
a second thread pool competing with its workers.
"""

from __future__ import annotations

import torch as _torch  # noqa: F401  (import order guard, see docstring)
import faiss as _faiss  # noqa: E402

_faiss.omp_set_num_threads(1)

__all__ = ["config"]

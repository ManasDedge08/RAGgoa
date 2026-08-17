"""Voice-enabled RAG over ai4bharat/MSMARCO-XI.

This module exists mostly to make the process survive its own dependencies and
its own host platform. Importing it first is what every entry point relies on.

**OpenMP.** torch, faiss and scikit-learn each ship their own copy of the
OpenMP runtime. Two consequences, both silent until they are not:

1. Importing faiss before torch makes the first CPU forward pass segfault on
   macOS/arm64, so the import order is pinned here rather than left to
   whichever module loads first.
2. faiss's first parallel region aborts the process with ``OMP: Error #15``
   when a second runtime is already initialised.

On macOS the second is fixed properly by ``scripts/fix_macos_openmp.py``, which
symlinks the duplicates to torch's copy. On Windows the runtimes are DLLs
inside each wheel and cannot be symlinked, so the documented escape hatch is
used instead — with faiss pinned to one thread, no second thread pool is
created and the "may silently produce incorrect results" warning does not apply
in practice. Linux wheels share the system OpenMP and need neither.

**Console encoding.** Windows still defaults stdout to a legacy code page, so
printing Hindi, Tamil or Bengali raises ``UnicodeEncodeError`` — which would
make every script in this repository crash on output rather than on anything
real. stdout and stderr are reconfigured to UTF-8 here.
"""

from __future__ import annotations

import os
import sys

if sys.platform == "win32":
    # Must be set before the OpenMP runtime initialises, i.e. before faiss.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    for _stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(_stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

import torch as _torch  # noqa: F401,E402  (import order guard, see docstring)
import faiss as _faiss  # noqa: E402

# A single-query HNSW search is serial regardless, and a web server should not
# have a second thread pool competing with its workers.
_faiss.omp_set_num_threads(1)

__all__ = ["config"]

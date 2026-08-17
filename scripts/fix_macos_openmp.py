"""Point every bundled libomp at one copy. macOS development machines only.

torch, faiss-cpu and scikit-learn each vendor their own ``libomp.dylib`` inside
their wheel. Loading two of them into one process aborts at faiss's first
parallel region:

    OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
    already initialized.

The documented escape hatch, ``KMP_DUPLICATE_LIB_OK=TRUE``, is described by
LLVM as able to silently produce incorrect results — not a trade worth making
for a search index. The actual fix is to have one runtime in the process, which
this script arranges by replacing the duplicates with symlinks to torch's copy.
Originals are kept alongside as ``libomp.dylib.bak`` so the change is
reversible.

Linux images do not need this: the manylinux wheels share the system OpenMP.

Run: ``python scripts/fix_macos_openmp.py``  (``--revert`` to undo)
"""

from __future__ import annotations

import platform
import sys
import sysconfig
from pathlib import Path

CANONICAL = "torch/lib/libomp.dylib"
DUPLICATES = ["faiss/.dylibs/libomp.dylib", "sklearn/.dylibs/libomp.dylib"]


def main() -> None:
    if platform.system() != "Darwin":
        print("not macOS; nothing to do")
        return

    site = Path(sysconfig.get_paths()["purelib"])
    canonical = site / CANONICAL
    if not canonical.exists():
        print(f"canonical runtime missing: {canonical}")
        sys.exit(1)

    revert = "--revert" in sys.argv
    for relative in DUPLICATES:
        path = site / relative
        backup = path.with_suffix(".dylib.bak")
        if revert:
            if backup.exists():
                path.unlink(missing_ok=True)
                backup.rename(path)
                print(f"restored {relative}")
            continue

        if path.is_symlink():
            print(f"already linked {relative}")
            continue
        if not path.exists():
            print(f"absent, skipping {relative}")
            continue
        path.rename(backup)
        path.symlink_to(canonical)
        print(f"linked {relative} -> {CANONICAL}")

    print("done" if not revert else "reverted")


if __name__ == "__main__":
    main()

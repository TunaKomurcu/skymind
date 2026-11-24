"""Hazır (prebuilt) bir DLL için MSVC içe aktarma kütüphanesi (.lib) üretir.

llama.cpp sürüm paketleri başlık ve .lib içermez; derlemek yerine hazır llama.dll'i bağlamak için:

    scripts\\cpp.cmd python scripts/make_import_lib.py third_party/llama-cuda/llama.dll

dumpbin ve lib.exe PATH'te olmalı (scripts\\cpp.cmd MSVC ortamını kurar).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

EXPORT_ROW = re.compile(r"^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+(\S+)")


def exports(dll: Path) -> list[str]:
    out = subprocess.run(
        ["dumpbin", "/exports", str(dll)], capture_output=True, text=True, check=True
    ).stdout
    names = [m.group(1) for line in out.splitlines() if (m := EXPORT_ROW.match(line))]
    if not names:
        raise SystemExit(f"{dll}: dışa aktarılan sembol bulunamadı")
    return names


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    dll = Path(sys.argv[1]).resolve()
    names = exports(dll)
    def_path = dll.with_suffix(".def")
    def_path.write_text(
        f"LIBRARY {dll.stem}\nEXPORTS\n" + "\n".join(f"    {n}" for n in names) + "\n",
        encoding="ascii",
    )
    lib_path = dll.with_suffix(".lib")
    subprocess.run(
        ["lib", "/nologo", "/machine:x64", f"/def:{def_path}", f"/out:{lib_path}"], check=True
    )
    print(f"{len(names)} sembol -> {lib_path}")


if __name__ == "__main__":
    main()

"""
Lightweight orjson compatibility shim for environments without orjson.
Provides minimal dumps/loads and OPT_INDENT_2 used in this repository's tests.
Note: orjson.dumps returns bytes; this shim mirrors that behavior.
"""
from __future__ import annotations
import json
from typing import Any, Union

# Options (subset)
OPT_INDENT_2 = 2


def dumps(obj: Any, option: int | None = None) -> bytes:
    indent = 2 if option == OPT_INDENT_2 else None
    # Ensure ASCII disabled to be closer to orjson (which supports UTF-8 bytes)
    s = json.dumps(obj, indent=indent, ensure_ascii=False)
    return s.encode("utf-8")


def loads(data: Union[str, bytes]) -> Any:
    if isinstance(data, (bytes, bytearray)):
        return json.loads(data.decode("utf-8"))
    return json.loads(data)

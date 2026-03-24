from os import PathLike
from typing import Any, BinaryIO, Sequence


def load(
    filename: str | PathLike[str] | BinaryIO,
    mmap_mode: str | None = None,
    ensure_native_byte_order: str = "auto",
) -> Any: ...


def dump(
    value: Any,
    filename: str | PathLike[str] | BinaryIO,
    compress: int | bool | tuple[str, int] = 0,
    protocol: int | None = None,
) -> Sequence[str]: ...

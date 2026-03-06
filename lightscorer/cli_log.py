from __future__ import annotations

from typing import Mapping, Optional


LINE = "═" * 72


def _emit(prefix: str, message: str) -> None:
    print(f"{prefix} {message}")


def banner(title: str) -> None:
    print(LINE)
    print(f"【{title}】")
    print(LINE)


def stage(index: int, total: int, title: str) -> None:
    print("")
    print(f"【阶段 {index}/{total}】{title}")


def info(message: str) -> None:
    _emit("[进行中]", message)


def success(message: str) -> None:
    _emit("[完成]", message)


def warn(message: str) -> None:
    _emit("[警告]", message)


def suggest(message: str) -> None:
    _emit("[建议]", message)


def key_values(title: str, data: Mapping[str, object], order: Optional[list[str]] = None) -> None:
    print(f"{title}:")
    keys = order if order is not None else list(data.keys())
    for k in keys:
        if k in data:
            print(f"  - {k}: {data[k]}")


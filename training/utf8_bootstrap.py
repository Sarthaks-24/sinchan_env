# Copyright (c) 2026. Shared Windows UTF-8 bootstrap for TRL / Transformers.
# TRL and related packages use Path.read_text() on UTF-8 .jinja files; the Windows
# default encoding (e.g. cp1252) raises UnicodeDecodeError (often at byte 0x9d).
"""Windows UTF-8 mode helpers for the training stack."""

from __future__ import annotations

import os
import sys

_REEXEC_ENVAR = "SINCHAN_WIN_UTF8_REEXEC"


def ensure_utf8_text_mode() -> None:
    """Re-execute the interpreter with ``-X utf8`` on Windows if not already in UTF-8 mode."""
    if os.name != "nt":
        return
    if bool(getattr(sys.flags, "utf8_mode", 0)):
        return
    if os.environ.get(_REEXEC_ENVAR) == "1":
        return
    os.environ[_REEXEC_ENVAR] = "1"
    try:
        os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])
    except OSError as exc:
        del os.environ[_REEXEC_ENVAR]
        print(
            f"[SINCHAN] Could not re-exec Python with -X utf8 ({exc!r}).\n"
            "  PowerShell:  $env:PYTHONUTF8='1'\n"
            "  Or run:      py -3 -X utf8 training/train_sinchan.py ...\n"
            "TRL/Transformers ship UTF-8 .jinja templates; the locale code page (cp1252) breaks them.\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def py_child_args() -> list[str]:
    """``subprocess`` prefix: use on Windows for child Python processes that may import TRL."""
    if os.name == "nt":
        return [sys.executable, "-X", "utf8"]
    return [sys.executable]

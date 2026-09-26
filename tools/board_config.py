"""Where the test board is on your network, for the laptop-side tools that talk to it over
WiFi: --host if given, else $STT_HOST, else `host = ...` in board.conf at the repo root
(git-ignored; copy board.conf.example). Nothing is built in.

    from board_config import board_host
    host = board_host(args.host)            # exits with a hint when nothing is configured
"""
import os
import sys
from pathlib import Path

CONF = Path(__file__).resolve().parent.parent / "board.conf"


def conf_value(key, path=CONF):
    """`key = value` from board.conf (# comments, blank lines ignored), or None."""
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        k, sep, v = line.split("#", 1)[0].partition("=")
        if sep and k.strip() == key:
            return v.strip() or None
    return None


def board_host(cli=None, required=True):
    host = cli or os.environ.get("STT_HOST") or conf_value("host")
    if host is None and required:
        sys.exit("board address unknown: pass --host, set STT_HOST, or put `host = <ip>` in "
                 f"{CONF.name} (see {CONF.name}.example)")
    return host

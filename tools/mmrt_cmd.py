#!/usr/bin/env python3
"""Send one text command to the stt firmware over TCP and print the JSON replies.

    tools/mmrt_cmd.py ktest
    tools/mmrt_cmd.py ref 1        # A/B: force portable C ops
    tools/mmrt_cmd.py info          # board address: $STT_HOST or board.conf
"""
import json
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_config import board_host  # noqa: E402

host = board_host()
s = socket.create_connection((host, 5555), timeout=900)
f = s.makefile("rb")
s.sendall((" ".join(sys.argv[1:]) + "\n").encode())
while True:
    line = f.readline().decode().strip()
    if not line:
        continue
    print(line)
    if line.startswith("DONE"):
        sys.exit(int(line.rsplit("rc=", 1)[1]) != 0)

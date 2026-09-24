#!/usr/bin/env python3
"""Send one text command to the stt firmware over TCP and print the JSON replies.

    tools/mmrt_cmd.py ktest
    tools/mmrt_cmd.py ref 1        # A/B: force portable C ops
    STT_HOST=192.168.1.154 tools/mmrt_cmd.py info
"""
import json
import os
import socket
import sys

host = os.environ.get("STT_HOST", "192.168.1.154")
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

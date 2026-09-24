# Which speech model is installed, and installing one from a file (SD card or internal
# filesystem) into the "model" flash partition, where stt reads it in place.
#
#   import model_tools
#   model_tools.info()                                   # {'format': 'int4', 'bytes': 5991232, ...}
#   model_tools.install("/sd/citrinet256_int4.mmrt")     # copies, verifies, resets the board
#
# The int4 model (6.0 MB) fits the standard layout's 6 MB partition; the int8 model
# (9.8 MB) needs the int8 layout (partition-table-int8.bin, 10 MB partition). The
# format is recorded per layer inside the file, so stt needs no setting either way.
import os

import esp32
import machine
import stt

CHUNK = 64 * 1024            # write in erase-aligned 64 KB steps


def partition():
    p = esp32.Partition.find(esp32.Partition.TYPE_DATA, label="model")
    if not p:
        raise OSError("no 'model' partition in this firmware's partition table")
    return p[0]


def info():
    """The installed model: format, size, and the partition it lives in."""
    stt.open()                                   # info() reports zeros until the model is open
    d = stt.info()
    p = partition()
    return {
        "format": "int4" if d["int4_ops"] else "int8",
        "bytes": d["model_bytes"],
        "int4_layers": d["int4_ops"],
        "partition_bytes": p.info()[3],
    }


def install(path, reset=True):
    """Copy an .mmrt file into the model partition, then check it byte for byte."""
    size = os.stat(path)[6]
    p = partition()
    cap = p.info()[3]
    with open(path, "rb") as f:
        if f.read(4) != b"MMRT":
            raise ValueError("%s is not an .mmrt model" % path)
    if size > cap:
        raise ValueError("model is %d bytes, the partition %d: use the int8 layout" % (size, cap))
    stt.close()                                  # the partition is memory-mapped while open
    buf = bytearray(CHUNK)
    blk = 0
    with open(path, "rb") as f:
        while True:
            n = f.readinto(buf)
            if not n:
                break
            if n < CHUNK:
                buf[n:] = b"\xff" * (CHUNK - n)
            p.writeblocks(blk, buf)
            blk += CHUNK // 4096
            print("\rinstalled %d / %d KB" % (min(blk * 4, size // 1024), size // 1024), end="")
    print()
    check = bytearray(CHUNK)
    with open(path, "rb") as f:                  # verify
        blk = 0
        while True:
            n = f.readinto(buf)
            if not n:
                break
            p.readblocks(blk, check)
            if check[:n] != buf[:n]:
                raise OSError("verify failed at block %d" % blk)
            blk += CHUNK // 4096
    print("verified %d bytes" % size)
    if reset:
        machine.reset()                          # stt maps the new model on next use

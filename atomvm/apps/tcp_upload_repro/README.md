# gen_tcp: long uploads end in `{error, closed}` partway through

A minimal repro for AtomVM on the ESP32-S3: a passive-mode `gen_tcp` server receiving
length-prefixed uploads of 100-330 KB. Occasionally `gen_tcp:recv(S, 0)` returns
`{error, closed}` in the middle of an upload, while the client is still sending.

Stock AtomVM only (`network`, `gen_tcp`, `io`): no NIFs, no custom components.

## Run it

1. `cp src/wifi_creds.erl.example src/wifi_creds.erl` and fill in your WiFi network.
2. Build and flash the app (exatomvm):

   ```bash
   mix deps.get && mix atomvm.packbeam
   esptool.py --chip esp32s3 write_flash 0x250000 tcp_upload_repro.avm   # stock release image layout
   ```

   `mix.exs` has `flash_offset: 0xA90000` for our own firmware layout; on a stock
   AtomVM release image `main.avm` is at `0x250000`.
3. The console prints `tcp_upload_repro listening on {a,b,c,d}:5556`. From a PC on the same
   network:

   ```bash
   python3 send_uploads.py <board-ip> -n 40
   ```

   It sends 40 uploads of random size (100-330 KB, 1 s apart) on one connection, reconnects
   after a failure, and prints `N/40 uploads failed` at the end.

## What we saw

On our firmware (AtomVM v0.7.0-alpha.1 + our NIF components, ESP32-S3 with octal PSRAM,
ESP-IDF 5.5), 2 of 34 uploads failed this way in a 4-pass soak, at random positions. The
board kept running; the next connection worked. That app did ~1.3 s of CPU-heavy work
(speech recognition, both cores) between uploads; this repro does none, to rule our code out.

**Not yet confirmed on a stock release image** -- this repro was written to check exactly
that. If it doesn't fail as is, try more load between uploads (a busy process on the board).

## What the board logs

Each upload: `conn C upload N: <bytes> bytes in <recv calls> recv calls, <ms> ms`.
A failure: `FAIL conn C upload N: closed after <received> of <expected> bytes (<recv calls>
recv calls, last data <ms> ms before), free heap <bytes>` -- how far it got, and whether the
data had stalled before the close.

## Server pattern (for reference)

Passive socket, `{active, false}`; one process per connection (`controlling_process`); the
reader loops `gen_tcp:recv(S, 0)`, gathering chunks in a list until it has the announced
length (recv returns whatever has arrived, often far less than asked).

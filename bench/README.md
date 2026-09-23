# bench — Phase 0 hardware measurements

One ESP-IDF app, one console command per benchmark. Results are printed as
`@@R {json}` lines; `tools/bench.py` collects them into `results/<date>-<board>.jsonl`.

## Build / flash

```sh
./idf.sh build
./idf.sh -p /dev/ttyACM0 flash          # add `monitor` for a raw console
```

`idf.sh` activates ESP-IDF v5.5.1 from `~/build/lvgl_micropython/lib/esp-idf`
(override with `IDF_PATH`). Targets the Waveshare S3 cam: 16MB flash, 8MB
**octal** PSRAM. For a quad-PSRAM board (T-Pager), switch `CONFIG_SPIRAM_MODE_OCT`
to `CONFIG_SPIRAM_MODE_QUAD` in `sdkconfig.defaults` and `rm sdkconfig` first.

Console is USB-Serial-JTAG, so the app shows up as `303a:1001`: the same VID:PID
as ROM download mode. Under WSL a single `usbipd bind` covers flashing and running
(unlike MicroPython's `303a:4001`).

## Run

```sh
tools/bench.py mem                  # headroom with ESP-IDF loaded
tools/bench.py membw [max_kb]       # SRAM/PSRAM bandwidth, hot vs stream
tools/bench.py flash [mb]           # mmapped-flash read bandwidth (model partition)
tools/bench.py sd [freq_khz] [mb]   # SD raw + FAT reads, 1-bit SDMMC
tools/bench.py fc [in] [out] [frames]   # int8 FC MAC throughput, weight reuse
tools/bench.py all
tools/bench.py --note "40MHz, sandisk 32GB" sd 40000
```

Every run appends to the day's jsonl with timestamp, board, command, git rev and
note, so results are comparable across firmware changes. Commit the jsonl files.

## Benchmarks

| cmd | measures | answers |
|---|---|---|
| `mem` | heaps (internal/DMA/PSRAM), app image vs partition, partition map, cache config | how much room is left once ESP-IDF is loaded |
| `membw` | read/write/copy, SRAM vs PSRAM, *hot* (same block) vs *stream* (walking 2MB) | cache vs PSRAM bus; cost of tiling weights into SRAM |
| `flash` | mmapped + `esp_partition_read` sequential reads of the 11MB `model` partition | flash vs SD as the expert store |
| `sd` | raw sectors + FAT file, 4/16/64KB chunks, into DMA SRAM vs PSRAM | expert swap time → routing granularity |
| `fc` | ESP-NN int8 FC: C vs PIE, weights PSRAM vs SRAM, per-frame vs row-major reuse, 1–16 frames | the active-param budget and the chunk size |

`sd` never formats. It needs a FAT card and writes one file, `/mmbench.bin`
(8MB by default). Raw reads come from the middle of the card and don't write.

## Adding a benchmark

1. `main/bench_<name>.c` with `int bench_<name>(int argc, char **argv)`, emitting
   results via `BENCH_RESULT("<name>", "\"k\":%d,...", ...)`.
2. Declare it in `main/bench.h`, add a row to `BENCHES[]` in `main/main.c`, add the
   file to `SRCS` in `main/CMakeLists.txt`.

## Queued

- SD / flash reads concurrent with `fc` on the other core (DMA overlap).
- Same app with WiFi + ESP-SR AFE linked, to re-run `mem` for real-firmware headroom.
- ESP-DL conv block from an exported `.espdl` model.
- AFE + VAD + log-mel CPU cost on the ES7210 dual-mic input.
- Bloom-filter connectivity layer kernel (IDEAS.md) vs int8 `fc`.

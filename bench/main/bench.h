// Benchmark harness. Each benchmark is a console command; results go out as
// one JSON object per line, prefixed "@@R ", for tools/bench.py to collect:
//
//   @@R {"bench":"membw","region":"psram","op":"read","bytes":1048576,"mbps":61.2}
//
// Adding a benchmark: write bench_<name>.c with `int bench_<name>(int argc,
// char **argv)`, declare it below, add it to the table in main.c and to SRCS in
// main/CMakeLists.txt.
#pragma once

#include <stdint.h>
#include <stdio.h>
#include "esp_timer.h"

#define BENCH_RESULT(name, fmt, ...) \
    printf("@@R {\"bench\":\"" name "\"," fmt "}\n", ##__VA_ARGS__)

static inline int64_t bench_now_us(void) { return esp_timer_get_time(); }

static inline double bench_mbps(size_t bytes, int64_t us)
{
    return us > 0 ? (double)bytes / (double)us : 0.0;  // bytes/us == MB/s
}

// Parse argv[i] as int, or return dflt when absent.
int bench_arg_int(int argc, char **argv, int i, int dflt);

int bench_mem(int argc, char **argv);
int bench_membw(int argc, char **argv);
int bench_flash(int argc, char **argv);
int bench_sd(int argc, char **argv);
int bench_fc(int argc, char **argv);
int bench_fc2(int argc, char **argv);
int bench_rec(int argc, char **argv);

// Console front end: one command per benchmark, plus `all`.
// Every run ends with "@@DONE <name> rc=<n>" so the host knows when to stop reading.
#include <stdlib.h>
#include <string.h>
#include "esp_console.h"
#include "esp_log.h"
#include "bench.h"

typedef struct {
    const char *name;
    const char *help;
    int (*fn)(int argc, char **argv);
} bench_t;

static const bench_t BENCHES[] = {
    {"mem",   "headroom: heaps, PSRAM, flash, app size vs partition", bench_mem},
    {"membw", "SRAM/PSRAM read/write/copy bandwidth [max_kb]", bench_membw},
    {"flash", "mmapped flash read bandwidth of the model partition [mb]", bench_flash},
    {"sd",    "SD read bandwidth, raw + FAT [freq_khz] [file_mb]", bench_sd},
    {"fc",    "int8 FC MAC throughput [in] [out] [frames]", bench_fc},
    {"fc2",   "fc row-reuse kernel on both cores vs one [in] [out] [frames]", bench_fc2},
};
#define N_BENCHES (sizeof(BENCHES) / sizeof(BENCHES[0]))

int bench_arg_int(int argc, char **argv, int i, int dflt)
{
    return i < argc ? atoi(argv[i]) : dflt;
}

static int run_one(const bench_t *b, int argc, char **argv)
{
    printf("@@START %s\n", b->name);
    int rc = b->fn(argc, argv);
    printf("@@DONE %s rc=%d\n", b->name, rc);
    return rc;
}

static int dispatch(int argc, char **argv)
{
    for (size_t i = 0; i < N_BENCHES; i++) {
        if (strcmp(argv[0], BENCHES[i].name) == 0) {
            return run_one(&BENCHES[i], argc, argv);
        }
    }
    return 1;
}

static int cmd_all(int argc, char **argv)
{
    int fails = 0;
    for (size_t i = 0; i < N_BENCHES; i++) {
        char *av[] = {(char *)BENCHES[i].name};
        fails += run_one(&BENCHES[i], 1, av) != 0;
    }
    printf("@@DONE all rc=%d\n", fails);
    return fails;
}

void app_main(void)
{
    esp_log_level_set("*", ESP_LOG_WARN);

    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t repl_cfg = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    repl_cfg.prompt = "bench>";
    repl_cfg.task_stack_size = 16 * 1024;
    esp_console_dev_usb_serial_jtag_config_t hw_cfg =
        ESP_CONSOLE_DEV_USB_SERIAL_JTAG_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_usb_serial_jtag(&hw_cfg, &repl_cfg, &repl));

    esp_console_register_help_command();
    for (size_t i = 0; i < N_BENCHES; i++) {
        const esp_console_cmd_t cmd = {
            .command = BENCHES[i].name,
            .help = BENCHES[i].help,
            .func = dispatch,
        };
        ESP_ERROR_CHECK(esp_console_cmd_register(&cmd));
    }
    const esp_console_cmd_t all = {.command = "all", .help = "run every benchmark with defaults",
                                   .func = cmd_all};
    ESP_ERROR_CHECK(esp_console_cmd_register(&all));

    printf("@@BOOT micromodels_bench\n");
    ESP_ERROR_CHECK(esp_console_start_repl(repl));
}

// Dink on an ESP32-S3: runs the example gadgets' requests and prints each answer next to the
// LLM's label and the PyTorch harness's answer, with timings; then takes typed requests.
//
//   ./idf.sh build
//   ./flash.sh                    # app + the model partition (models/dink-v13/dink.dnk)
//   ./idf.sh -p /dev/ttyACM0 monitor
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dink.h"
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "gadgets.h"
#include "mmrt.h"
#include "mmrt_s3.h"

static void *alloc16(size_t n) { return heap_caps_aligned_alloc(16, n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT); }
static void release(void *p) { heap_caps_free(p); }

static size_t free_internal(void) { return heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT); }

static void fmt(const demo_gadget_t *g, const dink_decision_t *r, char *out, size_t n)
{
    if (r->action < 0)
        snprintf(out, n, "(nothing)");
    else if (!g->actions[r->action].has_value)
        snprintf(out, n, "%s()", g->actions[r->action].name);
    else
        snprintf(out, n, "%s=%g", g->actions[r->action].name, r->value);
}

static int prepare(dink_t *d, const demo_gadget_t *dg, dink_gadget_t *g)
{
    *g = (dink_gadget_t){.fields = dg->fields, .n_fields = dg->n_fields, .actions = dg->actions, .n_actions = dg->n_actions};
    int64_t t0 = esp_timer_get_time();
    if (dink_gadget_prepare(d, g)) return -1;
    printf("  names encoded once per gadget: %d fields + %d actions in %lld ms\n", dg->n_fields, dg->n_actions,
           (esp_timer_get_time() - t0) / 1000);
    return 0;
}

static void run_demo(dink_t *d)
{
    int same = 0, right = 0, total = 0;
    int64_t sum_us = 0, max_us = 0;
    for (int gi = 0; gi < N_DEMO_GADGETS; gi++) {
        const demo_gadget_t *dg = &demo_gadgets[gi];
        printf("\n== %s\n", dg->setting);
        dink_gadget_t g;
        if (prepare(d, dg, &g)) {
            printf("prepare failed\n");
            return;
        }
        printf("  %-14s %-42s %-26s %-26s %s\n", "kind", "request", "LLM label", "Dink on the board", "ms");
        for (int i = 0; i < dg->n_requests; i++) {
            const demo_request_t *q = &dg->requests[i];
            dink_decision_t r;
            int64_t t0 = esp_timer_get_time();
            if (dink_decide(d, &g, q->say, &r)) {
                printf("decide failed\n");
                return;
            }
            int64_t us = esp_timer_get_time() - t0;
            char got[64];
            fmt(dg, &r, got, sizeof got);
            int s = !strcmp(got, q->torch), ok = !strcmp(got, q->llm);
            same += s;
            right += ok;
            total++;
            sum_us += us;
            max_us = us > max_us ? us : max_us;
            printf("  %-14s %-42.42s %-26.26s %-26.26s %3lld  %s%s\n", q->kind, q->say, q->llm, got, us / 1000,
                   ok ? "=LLM" : "    ", s ? "" : "  (PyTorch: different)");
            if (!s) printf("  %59s PyTorch: %s\n", "", q->torch);
        }
        dink_gadget_free(d, &g);
    }
    printf("\n%d requests: %d exactly match the LLM's label, %d match PyTorch on the PC.\n", total, right, same);
    printf("time per request (both passes, request encoding included): mean %lld ms, max %lld ms\n",
           sum_us / total / 1000, max_us / 1000);
}

static void timing_breakdown(dink_t *d)
{
    dink_prompt_t p;
    dink_name_t n;
    int64_t t0 = esp_timer_get_time();
    for (int i = 0; i < 10; i++) dink_encode_prompt(d, "make the water a little warmer please", &p);
    int64_t enc = (esp_timer_get_time() - t0) / 10;
    t0 = esp_timer_get_time();
    for (int i = 0; i < 10; i++) dink_encode_name(d, "set_heater_temperature", &n);
    int64_t nm = (esp_timer_get_time() - t0) / 10;
    // 8 state fields + 8 options: the largest scoring pass
    dink_item_t st[8], op[8];
    for (int i = 0; i < 8; i++) {
        st[i] = (dink_item_t){&n, 50.0f + i, 0, 100, 1};
        op[i] = (dink_item_t){&n, 10.0f * i, 0, 100, 1};
    }
    float probs[9];
    t0 = esp_timer_get_time();
    for (int i = 0; i < 10; i++) dink_score(d, st, 8, &p, op, 8, probs);
    int64_t sc = (esp_timer_get_time() - t0) / 10;
    printf("\ntiming: encode a request %lld us, encode a name %lld us, score 8 options against 8 fields %lld us\n",
           enc, nm, sc);

    // the S3 vector kernels against MMRT's portable reference, on Dink's trunk
    dink_prompt_t a, b;
    memset(&a, 0, sizeof a);                       // unused number slots stay zero in both
    memset(&b, 0, sizeof b);
    mmrt_use_ref = 1;
    t0 = esp_timer_get_time();
    dink_encode_prompt(d, "make the water a little warmer please", &a);
    int64_t ref_us = esp_timer_get_time() - t0;
    mmrt_use_ref = 0;
    dink_encode_prompt(d, "make the water a little warmer please", &b);
    printf("S3 kernels vs portable reference: %s (reference encode %lld us)\n",
           memcmp(&a, &b, sizeof a) ? "DIFFERENT" : "identical", ref_us);
}

static void interactive(dink_t *d)
{
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    usb_serial_jtag_driver_install(&cfg);
    usb_serial_jtag_vfs_use_driver();
    setvbuf(stdin, NULL, _IONBF, 0);
    int gi = 0;
    dink_gadget_t g;
    prepare(d, &demo_gadgets[gi], &g);
    printf("\nType a request for \"%s\" (:g N switches gadget, 0..%d)\n", demo_gadgets[gi].setting, N_DEMO_GADGETS - 1);
    char line[128];
    for (;;) {
        printf("dink> ");
        fflush(stdout);
        size_t n = 0;
        for (;;) {                                     // read a line, echoing it
            int c = fgetc(stdin);
            if (c == EOF) {
                vTaskDelay(1);
                continue;
            }
            if (c == '\r' || c == '\n') break;
            if (n < sizeof line - 1) line[n++] = (char)c;
        }
        line[n] = 0;
        printf("%s\n", line);
        if (!n) continue;
        if (!strncmp(line, ":g ", 3)) {
            int k = atoi(line + 3);
            if (k >= 0 && k < N_DEMO_GADGETS) {
                dink_gadget_free(d, &g);
                gi = k;
                prepare(d, &demo_gadgets[gi], &g);
                printf("gadget: %s\n", demo_gadgets[gi].setting);
            }
            continue;
        }
        dink_decision_t r;
        int64_t t0 = esp_timer_get_time();
        dink_decide(d, &g, line, &r);
        char got[64];
        fmt(&demo_gadgets[gi], &r, got, sizeof got);
        printf("  -> %s  %.0f%%%s  (%lld ms, %d pass%s)\n", got, r.p * 100, r.unsure ? "  unsure" : "",
               (esp_timer_get_time() - t0) / 1000, r.passes, r.passes > 1 ? "es" : "");
    }
}

void app_main(void)
{
    printf("\nDink on %s\n", CONFIG_IDF_TARGET);
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    const void *img = NULL;
    esp_partition_mmap_handle_t mh;
    if (!part || esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &img, &mh) != ESP_OK) {
        printf("no model partition: flash models/dink-v13/dink.dnk at 0x210000\n");
        return;
    }
    const uint32_t *h = (const uint32_t *)img;
    size_t size = h[6] + h[7];
    if (h[0] != 0x314B4E44u || size > part->size) {
        printf("the model partition holds no Dink model (flash dink.dnk at 0x210000)\n");
        return;
    }
    mmrt_s3_init();
    size_t before = free_internal();
    int64_t t0 = esp_timer_get_time();
    dink_t *d = dink_open(img, size, alloc16, release);
    if (!d) {
        printf("dink_open failed\n");
        return;
    }
    printf("model %u KB, read in place from flash; opened in %lld us, working memory %u KB; internal RAM free %u KB\n",
           (unsigned)(size / 1024), esp_timer_get_time() - t0, (unsigned)((before - free_internal()) / 1024),
           (unsigned)(free_internal() / 1024));
    run_demo(d);
    timing_breakdown(d);
    interactive(d);
}

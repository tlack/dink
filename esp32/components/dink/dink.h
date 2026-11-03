// Dink: a 204K-parameter decision model (https://github.com/tlack/dink-model) on MMRT.
//
// Given a gadget's state (named numbers with ranges), a short request and up to 8 offered
// options, Dink gives each option a probability, plus one for "nothing relevant". The text
// encoder's conv trunk runs on MMRT in int8 (BabyTalk's runtime: the speech model's kernels);
// the rest (pooling, value features, the relation step and a 2-layer attention mixer) is
// float32, on MMRT's mmrt_f32 ops.
//
// The model file (.dnk, from dink/device/export.py) is read in place, e.g. from a
// memory-mapped flash partition; dink_open allocates 59 KB of working memory.
//
// Two levels of API:
//   dink_decide()  the harness a device uses: a gadget (state + actions) and a request in, one
//                  action and value out. It builds the options itself, in two passes (which
//                  action and which way, then how much), as dink-model's dink/harness.py.
//   dink_score()   the network alone: score options you chose.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DINK_SLOTS 8          // state fields, and separately options
#define DINK_DIM 64
#define DINK_NAME_LEN 40      // bytes of a name the model reads (longer names keep their end)
#define DINK_PROMPT_LEN 64    // bytes of a request
#define DINK_NUMBERS 3        // numbers read from a request

typedef void *(*dink_alloc_fn)(size_t bytes);
typedef void (*dink_free_fn)(void *p);

typedef struct dink dink_t;

// Bind to a model image (not copied; it must outlive the handle). alloc/release serve MMRT's
// activations (8 KB each at most) and the working memory (59 KB measured): both must return
// 16-byte aligned memory (MMRT's vector kernels), e.g. heap_caps_aligned_alloc(16, ...).
// Returns NULL on a bad image.
dink_t *dink_open(const void *image, size_t size, dink_alloc_fn alloc, dink_free_fn release);
void dink_close(dink_t *d);

// ---- the network ----

// A name's encoding: its vector and the relation step's projections of it. Names are fixed
// for a device, so encode them once and keep these.
typedef struct {
    float vec[DINK_DIM], q[DINK_DIM], k[DINK_DIM];
} dink_name_t;

// A request's encoding (4 pooled tokens); encode it once and score as many option sets as you
// like against it.
typedef struct {
    float tok[4][DINK_DIM];
    float num_val[DINK_NUMBERS];
    uint8_t num_level[DINK_NUMBERS];   // 1: a fraction of any range ("50 percent", "half")
    uint8_t n_num;
} dink_prompt_t;

typedef struct {
    const dink_name_t *name;
    float value, lo, hi;
    uint8_t arg;               // options: 0 = an action without a value (value/lo/hi ignored)
} dink_item_t;

int dink_encode_name(dink_t *d, const char *name, dink_name_t *out);
int dink_encode_prompt(dink_t *d, const char *prompt, dink_prompt_t *out);

// probs[n_options + 1]: one per option, "nothing relevant" last. n_state, n_options <= 8.
int dink_score(dink_t *d, const dink_item_t *state, int n_state, const dink_prompt_t *prompt,
               const dink_item_t *options, int n_options, float *probs);

// Numbers in a request, as dink-model's dink/data/numbers.py: "seventy two" -> 72 (raw, in
// the option's units), "50 percent" / "half" -> 0.5 (level). Returns how many (<= max).
int dink_numbers(const char *prompt, float *val, uint8_t *is_level, int max);

// ---- the harness ----

typedef struct {
    const char *name;
    float value, lo, hi;
} dink_field_t;

typedef struct {
    const char *name;
    float lo, hi;              // the value's range; has_value 0 for a plain action ("turn_on_ozone")
    uint8_t has_value;
    const char *sets;          // the state field it changes, or NULL
} dink_action_t;

// A gadget with its names encoded (dink_gadget_prepare): keep it while the gadget's fields
// and actions keep their names. Field values may change between requests.
typedef struct {
    dink_field_t *fields;      // up to 8 are shown to the model
    int n_fields;
    const dink_action_t *actions;
    int n_actions;
    dink_name_t *field_names;  // [n_fields]
    dink_name_t *action_names; // [n_actions]
    float (*action_rel)[DINK_DIM];  // [n_actions]: the name with '_' as ' ', to match requests
} dink_gadget_t;

int dink_gadget_prepare(dink_t *d, dink_gadget_t *g);   // allocates the encodings
void dink_gadget_free(dink_t *d, dink_gadget_t *g);

typedef struct {
    int action;                // index into actions, or -1 for "nothing relevant"
    float value;               // the value to set (actions with a value)
    float p;                   // Dink's probability for the pick
    uint8_t unsure;            // two different actions nearly tied: a device could ask which
    uint8_t passes;            // 1 or 2
} dink_decision_t;

int dink_decide(dink_t *d, const dink_gadget_t *g, const char *request, dink_decision_t *out);

#ifdef __cplusplus
}
#endif

// Dink's network: the spec is dink-model's dink/device/reference.py, step for step.
#include "dink.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "mmrt.h"
#include "mmrt_f32.h"

#if !defined(MMRT_API_VERSION) || MMRT_API_VERSION < 1
#error "Dink needs MMRT with mmrt_f32.h (MMRT_API_VERSION 1): update the esp32/babytalk submodule"
#endif

#define DNK_MAGIC 0x314B4E44u   // "DNK1"
#define WIDTH 128
#define EMB 32
#define PTOK 4
#define HEADS 4
#define FF 256
#define MAXTOK (DINK_SLOTS + PTOK + DINK_SLOTS + 1)
#define BINS 16
#define REL_BINS 18

typedef struct {
    char name[24];
    uint32_t n, dims[4], off;
} entry_t;

typedef struct {
    const float *ln1_w, *ln1_b, *in_w, *in_b, *out_w, *out_b, *ln2_w, *ln2_b, *ff1_w, *ff1_b, *ff2_w, *ff2_b;
} layer_w_t;

typedef struct {
    const float *emb, *pos, *pool_u, *tnorm_w, *tnorm_b, *tout_w, *tout_b;
    const float *fuse0_w, *fuse0_b, *fuse1_w, *fuse1_b, *role, *nothing, *rel_q, *rel_k;
    const float *rel_out_w, *rel_out_b, *num_out_w, *num_out_b, *norm_w, *norm_b, *score_w, *score_b;
    layer_w_t m[2];
} weights_t;

struct dink {
    mmrt_model_t mm;
    weights_t w;
    dink_alloc_fn alloc;
    dink_free_fn release;
    int e_in, e_out;                       // the trunk's input / output exponents
    // working memory
    int8_t x0[DINK_PROMPT_LEN * EMB] __attribute__((aligned(16)));   // MMRT's input: 16-byte aligned
    float h[MAXTOK * DINK_DIM], hn[MAXTOK * DINK_DIM], qkv[MAXTOK * 3 * DINK_DIM], att[MAXTOK * DINK_DIM];
    float ff[MAXTOK * FF];
    float items[2 * DINK_SLOTS * (DINK_DIM + BINS + 1)];
};

static const float REL_EDGES[REL_BINS + 1] = {-1.0f, -0.65f, -0.45f, -0.3f, -0.2f, -0.15f, -0.1f, -0.05f, -0.02f,
                                              0.0f,  0.02f,  0.05f,  0.1f,  0.15f, 0.2f,   0.3f,   0.45f, 0.65f, 1.0f};

// ---- helpers (the dense float32 ops are MMRT's: mmrt_f32.h) ----

static float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---- open ----

static const float *find(const uint8_t *img, const uint32_t *hdr, const char *name, uint32_t n)
{
    const entry_t *e = (const entry_t *)(img + hdr[3]);
    for (uint32_t i = 0; i < hdr[2]; i++)
        if (strncmp(e[i].name, name, sizeof(e[i].name)) == 0)
            return e[i].n == n ? (const float *)(img + hdr[6] + e[i].off) : NULL;
    return NULL;
}

dink_t *dink_open(const void *image, size_t size, dink_alloc_fn alloc, dink_free_fn release)
{
    const uint8_t *img = (const uint8_t *)image;
    const uint32_t *h = (const uint32_t *)image;
    if (size < 64 || h[0] != DNK_MAGIC || h[1] != 1 || h[6] + h[7] > size || h[4] + h[5] > size) return NULL;
    if (h[8] != DINK_NAME_LEN || h[9] != DINK_PROMPT_LEN || h[10] != PTOK || h[11] != DINK_SLOTS ||
        h[12] != DINK_NUMBERS || h[13] != DINK_DIM || h[14] != WIDTH || h[15] != EMB)
        return NULL;
    dink_t *d = (dink_t *)alloc(sizeof(dink_t));
    if (!d) return NULL;
    memset(d, 0, sizeof(*d));
    d->alloc = alloc;
    d->release = release;
    if (mmrt_open(&d->mm, img + h[4], h[5]) != 0) {
        release(d);
        return NULL;
    }
    d->e_in = d->mm.tensors[d->mm.hdr->input].exp;
    d->e_out = d->mm.tensors[d->mm.hdr->output].exp;
    weights_t *w = &d->w;
    int ok = 1;
#define GET(field, nm, n) ok &= (w->field = find(img, h, nm, n)) != NULL
    GET(emb, "emb", 256 * EMB);
    GET(pos, "pos", DINK_PROMPT_LEN * EMB);
    GET(pool_u, "pool_u", PTOK * WIDTH);
    GET(tnorm_w, "tnorm_w", WIDTH);
    GET(tnorm_b, "tnorm_b", WIDTH);
    GET(tout_w, "tout_w", DINK_DIM * WIDTH);
    GET(tout_b, "tout_b", DINK_DIM);
    GET(fuse0_w, "fuse0_w", 128 * (DINK_DIM + BINS + 1));
    GET(fuse0_b, "fuse0_b", 128);
    GET(fuse1_w, "fuse1_w", DINK_DIM * 128);
    GET(fuse1_b, "fuse1_b", DINK_DIM);
    GET(role, "role", 3 * DINK_DIM);
    GET(nothing, "nothing", DINK_DIM);
    GET(rel_q, "rel_q", DINK_DIM * DINK_DIM);
    GET(rel_k, "rel_k", DINK_DIM * DINK_DIM);
    GET(rel_out_w, "rel_out_w", DINK_DIM * (REL_BINS + 2));
    GET(rel_out_b, "rel_out_b", DINK_DIM);
    GET(num_out_w, "num_out_w", DINK_DIM * DINK_NUMBERS * 5);
    GET(num_out_b, "num_out_b", DINK_DIM);
    GET(norm_w, "norm_w", DINK_DIM);
    GET(norm_b, "norm_b", DINK_DIM);
    GET(score_w, "score_w", DINK_DIM);
    GET(score_b, "score_b", 1);
    for (int i = 0; i < 2; i++) {
        char nm[24];
#define GETM(field, n) (snprintf(nm, sizeof nm, "m%d_" #field, i), ok &= (w->m[i].field = find(img, h, nm, n)) != NULL)
        GETM(ln1_w, DINK_DIM);
        GETM(ln1_b, DINK_DIM);
        GETM(in_w, 3 * DINK_DIM * DINK_DIM);
        GETM(in_b, 3 * DINK_DIM);
        GETM(out_w, DINK_DIM * DINK_DIM);
        GETM(out_b, DINK_DIM);
        GETM(ln2_w, DINK_DIM);
        GETM(ln2_b, DINK_DIM);
        GETM(ff1_w, FF * DINK_DIM);
        GETM(ff1_b, FF);
        GETM(ff2_w, DINK_DIM * FF);
        GETM(ff2_b, DINK_DIM);
    }
    if (!ok) {
        dink_close(d);
        return NULL;
    }
    return d;
}

void dink_close(dink_t *d)
{
    if (!d) return;
    mmrt_close(&d->mm, d->release);
    d->release(d);
}

// ---- text encoder ----

// out[queries][64]. The text's bytes: UTF-8, zero padded to L; a longer text keeps its end.
static int text(dink_t *d, const char *s, int L, int queries, float *out)
{
    uint8_t b[DINK_PROMPT_LEN] = {0};
    size_t n = strlen(s);
    const char *src = n > (size_t)L ? s + n - L : s;
    memcpy(b, src, n > (size_t)L ? (size_t)L : n);
    const weights_t *w = &d->w;
    const float sc = ldexpf(1.0f, -d->e_in);
    for (int l = 0; l < L; l++)
        for (int c = 0; c < EMB; c++) {
            float v = floorf((w->emb[b[l] * EMB + c] + w->pos[l * EMB + c]) * sc + 0.5f);
            d->x0[l * EMB + c] = (int8_t)clampf(v, -128, 127);
        }
    int T;
    const int8_t *y = mmrt_run(&d->mm, d->x0, L, &T, d->alloc, d->release);
    if (!y || T != L) return -1;
    const float so = ldexpf(1.0f, d->e_out);     // y[l][c] * so is the trunk's output
    float att[DINK_PROMPT_LEN], pooled[WIDTH], hn[PTOK * WIDTH];
    for (int q = 0; q < queries; q++) {
        const float *u = w->pool_u + q * WIDTH;
        for (int l = 0; l < L; l++) {
            float acc = 0;
            for (int c = 0; c < WIDTH; c++) acc += u[c] * y[l * WIDTH + c];
            att[l] = b[l] ? acc * so : -1e4f;
        }
        mmrt_f32_softmax(att, L);
        for (int c = 0; c < WIDTH; c++) pooled[c] = 0;
        for (int l = 0; l < L; l++)
            for (int c = 0; c < WIDTH; c++) pooled[c] += att[l] * so * y[l * WIDTH + c];
        mmrt_f32_layernorm(pooled, w->tnorm_w, w->tnorm_b, 1, WIDTH, 1e-5f, hn + q * WIDTH);
    }
    mmrt_f32_linear(w->tout_w, w->tout_b, hn, queries, DINK_DIM, WIDTH, out);
    return 0;
}

int dink_encode_name(dink_t *d, const char *name, dink_name_t *out)
{
    if (text(d, name, DINK_NAME_LEN, 1, out->vec)) return -1;
    mmrt_f32_linear(d->w.rel_q, NULL, out->vec, 1, DINK_DIM, DINK_DIM, out->q);
    mmrt_f32_linear(d->w.rel_k, NULL, out->vec, 1, DINK_DIM, DINK_DIM, out->k);
    return 0;
}

int dink_encode_prompt(dink_t *d, const char *prompt, dink_prompt_t *out)
{
    if (text(d, prompt, DINK_PROMPT_LEN, PTOK, &out->tok[0][0])) return -1;
    out->n_num = (uint8_t)dink_numbers(prompt, out->num_val, out->num_level, DINK_NUMBERS);
    return 0;
}

// ---- scoring ----

static float level(float v, float lo, float hi)
{
    return hi > lo ? clampf((v - lo) / (hi - lo), 0, 1) : 0.0f;
}

// item tokens for n items at once: fuse([name vec | 16 bins of level * has | has]) + role
static void fuse_rows(dink_t *d, const float *const *vec, const float *lvl, const float *has, const int *role,
                      int n, float *y)
{
    const int F = DINK_DIM + BINS + 1;
    for (int j = 0; j < n; j++) {
        float *f = d->items + j * F;
        memcpy(f, vec[j], sizeof(float) * DINK_DIM);
        for (int i = 0; i < BINS; i++) f[DINK_DIM + i] = clampf((lvl[j] - (float)i / BINS) * BINS, 0, 1) * has[j];
        f[DINK_DIM + BINS] = has[j];
    }
    mmrt_f32_linear(d->w.fuse0_w, d->w.fuse0_b, d->items, n, 128, F, d->ff);
    mmrt_f32_relu(d->ff, n * 128);
    mmrt_f32_linear(d->w.fuse1_w, d->w.fuse1_b, d->ff, n, DINK_DIM, 128, y);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < DINK_DIM; i++) y[j * DINK_DIM + i] += d->w.role[role[j] * DINK_DIM + i];
}

static void mixer_layer(dink_t *d, int n, int li)
{
    const int D = DINK_DIM;
    const layer_w_t *m = &d->w.m[li];
    mmrt_f32_layernorm(d->h, m->ln1_w, m->ln1_b, n, D, 1e-5f, d->hn);
    mmrt_f32_linear(m->in_w, m->in_b, d->hn, n, 3 * D, D, d->qkv);
    mmrt_f32_attention(d->qkv, n, D, HEADS, d->att);
    mmrt_f32_linear(m->out_w, m->out_b, d->att, n, D, D, d->hn);
    for (int i = 0; i < n * D; i++) d->h[i] += d->hn[i];
    mmrt_f32_layernorm(d->h, m->ln2_w, m->ln2_b, n, D, 1e-5f, d->hn);
    mmrt_f32_linear(m->ff1_w, m->ff1_b, d->hn, n, FF, D, d->ff);
    mmrt_f32_relu(d->ff, n * FF);
    mmrt_f32_linear(m->ff2_w, m->ff2_b, d->ff, n, D, FF, d->att);
    for (int i = 0; i < n * D; i++) d->h[i] += d->att[i];
}

int dink_score(dink_t *d, const dink_item_t *state, int n_state, const dink_prompt_t *prompt,
               const dink_item_t *options, int n_options, float *probs)
{
    if (n_state < 0 || n_state > DINK_SLOTS || n_options < 1 || n_options > DINK_SLOTS) return -1;
    const int D = DINK_DIM;
    const weights_t *w = &d->w;
    float s_lvl[DINK_SLOTS];
    // the item tokens, state then options, in one batch; placed around the prompt's tokens
    const float *vec[2 * DINK_SLOTS];
    float lv[2 * DINK_SLOTS], has[2 * DINK_SLOTS];
    int role[2 * DINK_SLOTS];
    for (int i = 0; i < n_state; i++) {
        s_lvl[i] = level(state[i].value, state[i].lo, state[i].hi);
        vec[i] = state[i].name->vec, lv[i] = s_lvl[i], has[i] = 1.0f, role[i] = 0;
    }
    for (int j = 0; j < n_options; j++) {
        const dink_item_t *o = &options[j];
        const int k = n_state + j;
        vec[k] = o->name->vec, role[k] = 1;
        has[k] = o->arg ? 1.0f : 0.0f;
        lv[k] = o->arg ? level(o->value, o->lo, o->hi) : 0.0f;
    }
    float *items = d->att;                     // free until the mixer
    fuse_rows(d, vec, lv, has, role, n_state + n_options, items);
    memcpy(d->h, items, sizeof(float) * D * n_state);
    int t = n_state;
    for (int q = 0; q < PTOK; q++, t++)
        for (int i = 0; i < D; i++) d->h[t * D + i] = prompt->tok[q][i] + w->role[2 * D + i];
    for (int j = 0; j < n_options; j++, t++) {
        const dink_item_t *o = &options[j];
        const int arg = o->arg != 0;
        const float lo = arg ? o->lo : 0.0f, hi = arg ? o->hi : 1.0f;
        const float lvl = arg ? level(o->value, lo, hi) : 0.0f;
        float *tok = d->h + t * D;
        memcpy(tok, items + (n_state + j) * D, sizeof(float) * D);
        // relation step: which state field this option changes, and by how much
        float attended = 0, match = 0;
        if (n_state) {
            float a[DINK_SLOTS];
            for (int s = 0; s < n_state; s++) {
                float acc = 0;
                for (int i = 0; i < D; i++) acc += o->name->q[i] * state[s].name->k[i];
                a[s] = acc / 8.0f;
            }
            mmrt_f32_softmax(a, n_state);
            for (int s = 0; s < n_state; s++) {
                attended += a[s] * s_lvl[s];
                match = a[s] > match ? a[s] : match;
            }
        }
        const float delta = arg ? lvl - attended : 0.0f;
        float rel[REL_BINS + 2], tmp[DINK_DIM];
        for (int i = 0; i < REL_BINS; i++)
            rel[i] = clampf((delta - REL_EDGES[i]) / (REL_EDGES[i + 1] - REL_EDGES[i]), 0, 1);
        rel[REL_BINS] = fabsf(delta);
        rel[REL_BINS + 1] = match;
        mmrt_f32_linear(w->rel_out_w, w->rel_out_b, rel, 1, D, REL_BINS + 2, tmp);
        for (int i = 0; i < D; i++) tok[i] += tmp[i];
        // numbers from the request, each read in this option's own range
        float nf[DINK_NUMBERS * 5] = {0};
        if (arg)
            for (int k = 0; k < prompt->n_num; k++) {
                const float nv = prompt->num_val[k];
                const float l = prompt->num_level[k] ? nv : (nv - lo) / fmaxf(hi - lo, 1e-9f);
                const float inr = (l >= -1e-6f && l <= 1 + 1e-6f) ? 1.0f : 0.0f;
                const float dist = fabsf(lvl - clampf(l, 0, 1));
                float *f = nf + k * 5;
                f[0] = 1.0f;
                f[1] = inr;
                f[2] = clampf(l, -1, 2) / 2;
                f[3] = dist;
                f[4] = (dist < 1e-3f ? 1.0f : 0.0f) * inr;
            }
        mmrt_f32_linear(w->num_out_w, w->num_out_b, nf, 1, D, DINK_NUMBERS * 5, tmp);
        for (int i = 0; i < D; i++) tok[i] += tmp[i];
    }
    memcpy(d->h + t * D, w->nothing, sizeof(float) * D);
    const int n = ++t;
    for (int li = 0; li < 2; li++) mixer_layer(d, n, li);
    const int first = n_state + PTOK;
    float logit[DINK_SLOTS + 1], hn[DINK_DIM];
    for (int j = 0; j <= n_options; j++) {
        mmrt_f32_layernorm(d->h + (first + j) * D, w->norm_w, w->norm_b, 1, D, 1e-5f, hn);
        float acc = w->score_b[0];
        for (int i = 0; i < D; i++) acc += w->score_w[i] * hn[i];
        logit[j] = acc;
    }
    mmrt_f32_softmax(logit, n_options + 1);
    memcpy(probs, logit, sizeof(float) * (n_options + 1));
    return 0;
}

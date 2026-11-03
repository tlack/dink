// Numbers in a request and the two-pass harness: dink-model's dink/data/numbers.py and
// dink/harness.py, ported line for line (Python's round-half-even is rint() here).
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dink.h"

// ---- numbers ----

static const char *ONES[] = {"zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                             "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                             "seventeen", "eighteen", "nineteen"};
static const int ONES_V[] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
static const char *TENS[] = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};

#define MAXTOK 32
#define TOKLEN 24

typedef struct {
    char t[MAXTOK][TOKLEN];
    int n;
} toks_t;

// the token without leading/trailing characters from `chars`
static void strip(const char *s, const char *chars, char *out)
{
    size_t a = 0, b = strlen(s);
    while (a < b && strchr(chars, s[a])) a++;
    while (b > a && strchr(chars, s[b - 1])) b--;
    memcpy(out, s + a, b - a);
    out[b - a] = 0;
}

static int lookup(const char *w, const char **tab, int n)
{
    for (int i = 0; i < n; i++)
        if (!strcmp(w, tab[i])) return i;
    return -1;
}

static int ones(const char *tok, const char *chars)   // value or -1
{
    char w[TOKLEN];
    strip(tok, chars, w);
    int i = lookup(w, ONES, 21);
    return i < 0 ? -1 : ONES_V[i];
}

static int tens(const char *tok, const char *chars)
{
    char w[TOKLEN];
    strip(tok, chars, w);
    int i = lookup(w, TENS, 8);
    return i < 0 ? -1 : 20 + 10 * i;
}

// "set it to seventy two" -> "set it to 72"; "one hundred" -> "100" (0..999)
static void words_to_digits(const char *text, char *out, size_t cap)
{
    toks_t tk = {.n = 0};
    const char *p = text;
    while (*p && tk.n < MAXTOK) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        int k = 0;
        while (*p && !isspace((unsigned char)*p)) {
            if (k < TOKLEN - 1) tk.t[tk.n][k++] = *p;
            p++;
        }
        tk.t[tk.n++][k] = 0;
    }
    out[0] = 0;
    size_t len = 0;
#define EMIT(s)                                                                 \
    do {                                                                        \
        size_t l_ = strlen(s);                                                  \
        if (len + l_ + 2 < cap) {                                               \
            if (len) out[len++] = ' ';                                          \
            memcpy(out + len, s, l_ + 1);                                       \
            len += l_;                                                          \
        }                                                                       \
    } while (0)
    int i = 0;
    while (i < tk.n) {
        char w[TOKLEN];
        strip(tk.t[i], ",.!?", w);
        int is_a = !strcmp(w, "a") && i + 1 < tk.n && !strcmp(tk.t[i + 1], "hundred");
        if (lookup(w, ONES, 21) >= 0 || lookup(w, TENS, 8) >= 0 || is_a) {
            int n = 0, j = i;
            char h[TOKLEN];
            if (j + 1 < tk.n) strip(tk.t[j + 1], ",.", h); else h[0] = 0;
            if ((ones(tk.t[j], ",.") >= 0 && j + 1 < tk.n && !strcmp(h, "hundred")) || !strcmp(tk.t[j], "a")) {
                int o = ones(tk.t[j], ",.");
                n += 100 * (o >= 0 ? o : 1);
                j += 2;
                if (j < tk.n && !strcmp(tk.t[j], "and")) j++;
            }
            if (j < tk.n && tens(tk.t[j], ",.") >= 0) {
                n += tens(tk.t[j], ",.");
                j++;
                int o = j < tk.n ? ones(tk.t[j], ",.") : -1;
                if (o >= 0 && o < 10) {
                    n += o;
                    j++;
                }
            } else if (j < tk.n && ones(tk.t[j], ",.") >= 0) {
                n += ones(tk.t[j], ",.");
                j++;
            }
            if (j > i) {
                char num[16];
                snprintf(num, sizeof num, "%d", n);
                EMIT(num);
                i = j;
                continue;
            }
        }
        EMIT(tk.t[i]);
        i++;
    }
#undef EMIT
}

static int is_word(char c) { return isalnum((unsigned char)c) || c == '_'; }

// regex \bword\b in text
static int has_word(const char *text, const char *word)
{
    size_t n = strlen(word);
    for (const char *p = strstr(text, word); p; p = strstr(p + 1, word))
        if ((p == text || !is_word(p[-1])) && !is_word(p[n])) return 1;
    return 0;
}

// regex \b(to|at|on)\s+word\b
static int has_setting(const char *text, const char *word)
{
    static const char *preps[] = {"to", "at", "on"};
    for (int i = 0; i < 3; i++)
        for (const char *p = strstr(text, preps[i]); p; p = strstr(p + 1, preps[i])) {
            if (p != text && is_word(p[-1])) continue;
            const char *q = p + 2;
            if (!isspace((unsigned char)*q)) continue;
            while (isspace((unsigned char)*q)) q++;
            size_t n = strlen(word);
            if (!strncmp(q, word, n) && !is_word(q[n])) return 1;
        }
    return 0;
}

int dink_numbers(const char *prompt, float *val, uint8_t *is_level, int max)
{
    char low[256], text[320];
    size_t n = strlen(prompt);
    if (n > sizeof(low) - 1) n = sizeof(low) - 1;
    for (size_t i = 0; i < n; i++) low[i] = (char)tolower((unsigned char)prompt[i]);
    low[n] = 0;
    words_to_digits(low, text, sizeof text);
    int found = 0, any_level = 0;
    float fv[16];
    uint8_t fl[16];
    // -?\d+(?:\.\d+)?, leftmost matches
    for (const char *p = text; *p && found < 16;) {
        const char *s = p;
        if (*s == '-' && isdigit((unsigned char)s[1])) s++;
        if (!isdigit((unsigned char)*s)) {
            p++;
            continue;
        }
        const char *e = s;
        while (isdigit((unsigned char)*e)) e++;
        if (*e == '.' && isdigit((unsigned char)e[1])) {
            e++;
            while (isdigit((unsigned char)*e)) e++;
        }
        char num[32];                        // only the matched characters ("1e5" is 1, then "e5")
        size_t nlen = (size_t)(e - p) < sizeof num - 1 ? (size_t)(e - p) : sizeof num - 1;
        memcpy(num, p, nlen);
        num[nlen] = 0;
        double v = strtod(num, NULL);
        const char *rest = e;
        while (isspace((unsigned char)*rest)) rest++;
        int pct = *rest == '%' || !strncmp(rest, "percent", 7) || !strncmp(rest, "per cent", 8) ||
                  !strncmp(rest, "pct", 3);
        fv[found] = (float)(pct ? v / 100 : v);
        fl[found++] = (uint8_t)pct;
        any_level |= pct;
        p = e;
    }
    static const char *LW[] = {"half", "halfway", "quarter", "full", "max", "maximum", "minimum"};
    static const float LV[] = {0.5f, 0.5f, 0.25f, 1.0f, 1.0f, 1.0f, 0.0f};
    for (int i = 0; i < 7 && found < 16; i++) {
        if (any_level || !has_word(text, LW[i])) continue;
        if (i >= 3 && !has_setting(text, LW[i])) continue;   // "full" alone is often not a setting
        fv[found] = LV[i];
        fl[found++] = 1;
        any_level = 1;
    }
    int k = found < max ? found : max;
    for (int i = 0; i < k; i++) {
        val[i] = fv[i];
        is_level[i] = fl[i];
    }
    return k;
}

// ---- harness ----

#define TIE 0.15f
static const double FRACTIONS[4] = {0.15, 0.35, 0.6, 1.0};

static double round_to(double v, int places)  // Python's round(v, places), half to even
{
    double m = pow(10, places);
    return rint(v * m) / m;
}

// the ordinal generator's simple values (dink-model's dink/gen/ordinal.py: nice)
static double nice(double v, double lo, double hi, int whole)
{
    double span = hi - lo, step;
    if (whole || span >= 10)
        step = span >= 50 ? 5 : 1;
    else
        step = 0.05 * span;
    if (hi - v < step / 2)
        v = hi;
    else if (v - lo < step / 2)
        v = lo;
    else
        v = rint(v / step) * step;
    v = v < lo ? lo : (v > hi ? hi : v);
    return (whole || span >= 10) && step >= 1 ? rint(v) : round_to(v, 2);
}

static int current(const dink_gadget_t *g, const dink_action_t *a, double *cur)
{
    if (!a->sets) return 0;
    for (int i = 0; i < g->n_fields; i++)
        if (!strcmp(g->fields[i].name, a->sets)) {
            *cur = g->fields[i].value;
            return 1;
        }
    return 0;
}

static int is_int(double v) { return v == floor(v); }

static int whole(const dink_action_t *a, int has_cur, double cur)
{
    return is_int(a->lo) && is_int(a->hi) && (!has_cur || is_int(cur));
}

// numbers in the request that fit this action's range (levels become values)
static int said(const dink_prompt_t *p, const dink_action_t *a, double *out)
{
    int n = 0;
    for (int k = 0; k < p->n_num; k++) {
        double v = p->num_val[k];
        if (p->num_level[k]) v = a->lo + v * ((double)a->hi - a->lo);
        if (a->lo <= v && v <= a->hi) out[n++] = round_to(v, 4);
    }
    return n;
}

static int contains(const double *v, int n, double x)
{
    for (int i = 0; i < n; i++)
        if (v[i] == x) return 1;
    return 0;
}

static float cosine(const float *a, const float *b)
{
    float ab = 0, aa = 0, bb = 0;
    for (int i = 0; i < DINK_DIM; i++) {
        ab += a[i] * b[i];
        aa += a[i] * a[i];
        bb += b[i] * b[i];
    }
    return ab / fmaxf(sqrtf(aa) * sqrtf(bb), 1e-8f);
}

int dink_gadget_prepare(dink_t *d, dink_gadget_t *g)
{
    g->field_names = (dink_name_t *)calloc(g->n_fields ? g->n_fields : 1, sizeof(dink_name_t));
    g->action_names = (dink_name_t *)calloc(g->n_actions, sizeof(dink_name_t));
    g->action_rel = calloc(g->n_actions, sizeof(*g->action_rel));
    if (!g->field_names || !g->action_names || !g->action_rel) return -1;
    for (int i = 0; i < g->n_fields; i++)
        if (dink_encode_name(d, g->fields[i].name, &g->field_names[i])) return -1;
    for (int i = 0; i < g->n_actions; i++) {
        char spaced[DINK_NAME_LEN * 4];
        snprintf(spaced, sizeof spaced, "%s", g->actions[i].name);
        for (char *c = spaced; *c; c++)
            if (*c == '_') *c = ' ';
        dink_name_t tmp;
        if (dink_encode_name(d, g->actions[i].name, &g->action_names[i]) || dink_encode_name(d, spaced, &tmp))
            return -1;
        memcpy(g->action_rel[i], tmp.vec, sizeof(tmp.vec));
    }
    return 0;
}

void dink_gadget_free(dink_t *d, dink_gadget_t *g)
{
    (void)d;
    free(g->field_names);
    free(g->action_names);
    free(g->action_rel);
    g->field_names = g->action_names = NULL;
    g->action_rel = NULL;
}

typedef struct {
    int action;
    double value;
} opt_t;

static void make_items(const dink_gadget_t *g, const opt_t *o, int n, dink_item_t *items)
{
    for (int i = 0; i < n; i++) {
        const dink_action_t *a = &g->actions[o[i].action];
        items[i] = (dink_item_t){&g->action_names[o[i].action], (float)o[i].value, a->lo, a->hi, a->has_value};
    }
}

int dink_decide(dink_t *d, const dink_gadget_t *g, const char *request, dink_decision_t *out)
{
    dink_prompt_t p;
    if (dink_encode_prompt(d, request, &p)) return -1;
    dink_item_t state[DINK_SLOTS], items[DINK_SLOTS];
    const int ns = g->n_fields < DINK_SLOTS ? g->n_fields : DINK_SLOTS;
    for (int i = 0; i < ns; i++)
        state[i] = (dink_item_t){&g->field_names[i], g->fields[i].value, g->fields[i].lo, g->fields[i].hi, 1};

    // pass 1: every action once; a valued action one step each way (35% of the room left), or
    // its range ends without a current value, plus numbers said that fit
    static opt_t groups[64][DINK_SLOTS + 4];
    int glen[64], na = g->n_actions < 64 ? g->n_actions : 64;
    for (int i = 0; i < na; i++) {
        const dink_action_t *a = &g->actions[i];
        glen[i] = 0;
        if (!a->has_value) {
            groups[i][glen[i]++] = (opt_t){i, 0};
            continue;
        }
        double cur = 0, vals[DINK_NUMBERS + 2];
        int has_cur = current(g, a, &cur), w = whole(a, has_cur, cur);
        int nv = said(&p, a, vals);
        if (!has_cur) {
            vals[nv++] = a->lo;
            vals[nv++] = a->hi;
        } else {
            double room_up = a->hi - cur, room_dn = cur - a->lo;
            if (room_up > 1e-9) vals[nv++] = nice(cur + 0.35 * room_up, a->lo, a->hi, w);
            if (room_dn > 1e-9) vals[nv++] = nice(cur - 0.35 * room_dn, a->lo, a->hi, w);
        }
        for (int k = 0; k < nv; k++) {       // dedupe, in order; never the current value
            if (has_cur && fabs(vals[k] - cur) <= 1e-9) continue;
            int dup = 0;
            for (int m = 0; m < glen[i]; m++) dup |= groups[i][m].value == vals[k];
            if (!dup) groups[i][glen[i]++] = (opt_t){i, vals[k]};
        }
    }
    int order[64], total = 0;
    for (int i = 0; i < na; i++) {
        order[i] = i;
        total += glen[i];
    }
    if (total > DINK_SLOTS) {               // keep the actions whose names match the request best
        float rel[64];
        for (int i = 0; i < na; i++) rel[i] = cosine(p.tok[0], g->action_rel[i]);
        for (int i = 1; i < na; i++)          // stable sort by relevance, best first
            for (int j = i; j > 0 && rel[order[j]] > rel[order[j - 1]]; j--) {
                int t = order[j];
                order[j] = order[j - 1];
                order[j - 1] = t;
            }
        int kept[64], nk = 0, n = 0;
        for (int i = 0; i < na; i++)
            if (n + glen[order[i]] <= DINK_SLOTS) {
                kept[nk++] = order[i];
                n += glen[order[i]];
            }
        // the Python keeps them in relevance order
        memcpy(order, kept, sizeof(int) * nk);
        na = nk;
    }
    opt_t opts[DINK_SLOTS] = {{0}};
    int no = 0;
    for (int i = 0; i < na; i++)
        for (int k = 0; k < glen[order[i]]; k++) opts[no++] = groups[order[i]][k];
    float probs[DINK_SLOTS + 1];
    make_items(g, opts, no, items);
    if (dink_score(d, state, ns, &p, items, no, probs)) return -1;
    int best = 0;
    for (int i = 1; i <= no; i++)
        if (probs[i] > probs[best]) best = i;
    out->passes = 1;
    out->unsure = 0;
    if (best == no) {
        out->action = -1;
        out->value = 0;
        out->p = probs[no];
        return 0;
    }
    // a near-tie between two different actions: flag it, and break it by name match
    int o1 = -1, o2 = -1;
    for (int i = 0; i < no; i++)
        if (o1 < 0 || probs[i] > probs[o1]) o1 = i;
    for (int i = 0; i < no; i++)
        if (i != o1 && (o2 < 0 || probs[i] > probs[o2])) o2 = i;
    opt_t pick = opts[best];
    float pp = probs[best];
    if (o2 >= 0 && opts[o2].action != pick.action && probs[o1] - probs[o2] < TIE) {
        if (cosine(p.tok[0], g->action_rel[opts[o2].action]) > cosine(p.tok[0], g->action_rel[opts[o1].action])) {
            pick = opts[o2];
            pp = probs[o2];
        }
        out->unsure = 1;
    }
    out->action = pick.action;
    out->value = (float)pick.value;
    out->p = pp;
    const dink_action_t *a = &g->actions[pick.action];
    double said_v[DINK_NUMBERS];
    int nsaid = a->has_value ? said(&p, a, said_v) : 0;
    if (!a->has_value || contains(said_v, nsaid, pick.value)) return 0;

    // pass 2: how much. A ladder both ways from the current value (15 / 35 / 60 / 100% of the
    // room left), numbers said first; evenly across the range without a current value.
    double cur = 0, vals[DINK_NUMBERS + 8], lad[DINK_NUMBERS + 8];
    int has_cur = current(g, a, &cur), w = whole(a, has_cur, cur), nv = 0, nl = 0;
    for (int k = 0; k < nsaid; k++) vals[nv++] = said_v[k];
    if (!has_cur) {
        for (int k = 0; k <= 4; k++) vals[nv++] = nice(a->lo + 0.25 * k * ((double)a->hi - a->lo), a->lo, a->hi, w);
    } else {
        for (int up = 1; up >= 0; up--) {
            double room = up ? a->hi - cur : cur - a->lo;
            for (int f = 0; f < 4; f++)
                if (room > 1e-9) vals[nv++] = nice(cur + (up ? 1 : -1) * FRACTIONS[f] * room, a->lo, a->hi, w);
        }
    }
    const double gap = 0.04 * ((double)a->hi - a->lo);
    for (int k = 0; k < nv; k++) {
        double v = vals[k];
        int keep = contains(said_v, nsaid, v);
        if (!keep && (!has_cur || fabs(v - cur) > 1e-9)) {
            keep = 1;
            for (int m = 0; m < nl; m++)
                if (fabs(v - lad[m]) < gap) keep = 0;
        }
        if (keep && !contains(lad, nl, v)) lad[nl++] = v;
    }
    if (nl > DINK_SLOTS) nl = DINK_SLOTS;
    if (nl <= 1) return 0;
    for (int k = 0; k < nl; k++) opts[k] = (opt_t){pick.action, lad[k]};
    make_items(g, opts, nl, items);
    if (dink_score(d, state, ns, &p, items, nl, probs)) return -1;
    int b2 = 0;
    for (int k = 1; k < nl; k++)                // the amount; "nothing" was ruled out in pass 1
        if (probs[k] > probs[b2]) b2 = k;
    out->value = (float)lad[b2];
    out->p = probs[b2];
    out->passes = 2;
    return 0;
}

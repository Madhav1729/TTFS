/* APEX PASC parameters per layer: integer theta and mem0 per channel that reproduce the threshold
 * table exactly (mismatches are counted).   apex_prep PROBLEM_FILE F OUT_PARAM_FILE */
#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int n, m, P, K;
static int8_t *wt; static int32_t *thr; static uint8_t *down, *lev;

static int64_t fdiv(int64_t a, int64_t b) { int64_t q = a / b; if ((a % b) && ((a < 0) != (b < 0))) --q; return q; }  /* floor */
static int64_t cdiv(int64_t a, int64_t b) { return -fdiv(-a, b); }

/* theta interval [lo, hi] (integers) for thresholds Tk[1..K] at scale S; returns 1 if non-empty */
static int theta_range(const int64_t *Tk, int64_t S, int64_t *lo, int64_t *hi) {
    int64_t l = 1, h = INT64_MAX / 4;
    for (int a = 1; a <= K; ++a) for (int b = a + 1; b <= K; ++b) {
        int64_t d = b - a;
        int64_t x = fdiv((Tk[b] - Tk[a] - 1) * S, d) + 1;      /* theta > (..)/d */
        int64_t y = cdiv((Tk[b] - Tk[a] + 1) * S, d) - 1;      /* theta < (..)/d */
        if (x > l) l = x;
        if (y < h) h = y;
    }
    *lo = l; *hi = h;
    return l <= h;
}

static int64_t mem0_for(const int64_t *Tk, int64_t S, int64_t th, int *hits) {
    int64_t lo = INT64_MIN, hi = INT64_MAX;
    for (int k = 1; k <= K; ++k) {
        int64_t a = k * th - Tk[k] * S, b = k * th - (Tk[k] - 1) * S;   /* a <= mem0 < b */
        if (a > lo) lo = a;
        if (b < hi) hi = b;
    }
    if (lo < hi) { *hits = K; return lo; }
    /* infeasible: the mem0 (from the candidate set of interval ends) that satisfies most k */
    int best = -1; int64_t bm = 0;
    for (int c = 1; c <= K; ++c) {
        int64_t cand = c * th - Tk[c] * S;
        int h = 0;
        for (int k = 1; k <= K; ++k) h += (cand >= k * th - Tk[k] * S) && (cand < k * th - (Tk[k] - 1) * S);
        if (h > best) { best = h; bm = cand; }
    }
    *hits = best; return bm;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: apex_prep PROBLEM F OUT\n"); return 2; }
    const int F = atoi(argv[2]);
    const int64_t S = (int64_t)1 << F;
    FILE *f = fopen(argv[1], "r"); assert(f);
    assert(fscanf(f, "%d %d %d %d", &n, &m, &P, &K) == 4);
    wt = malloc((size_t)m * n); thr = malloc(sizeof(int32_t) * (size_t)m * K); down = malloc(m); lev = malloc((size_t)P * n);
    for (size_t x = 0; x < (size_t)m * n; ++x) { int v; assert(fscanf(f, "%d", &v) == 1); assert(v >= -8 && v <= 7); wt[x] = (int8_t)v; }
    for (int j = 0; j < m; ++j) {
        for (int t = 0; t < K; ++t) assert(fscanf(f, "%d", &thr[(size_t)j * K + t]) == 1);
        int d; assert(fscanf(f, "%d", &d) == 1 && (d == 0 || d == 1)); down[j] = (uint8_t)d;
    }
    for (size_t x = 0; x < (size_t)P * n; ++x) { int v; assert(fscanf(f, "%d", &v) == 1); assert(v >= 0 && v <= K); lev[x] = (uint8_t)v; }
    fclose(f);

    int64_t *Tk = malloc(sizeof(int64_t) * (size_t)m * (K + 1));
    int64_t *theta = malloc(sizeof(int64_t) * m), *mem0 = malloc(sizeof(int64_t) * m);
    int inexact_ch = 0, nonmono = 0, fmin_layer = 0, min_hits = K;
    for (int j = 0; j < m; ++j) {
        int64_t *T_ = &Tk[(size_t)j * (K + 1)];
        for (int k = 1; k <= K; ++k) {
            int64_t e = down[j] ? INT64_MIN : INT64_MAX;
            for (int t = 0; t <= K - k; ++t) {
                int64_t v = thr[(size_t)j * K + t];
                if (down[j] ? v > e : v < e) e = v;
            }
            T_[k] = down[j] ? -e : e;
        }
        for (int t = 0; t + 1 < K; ++t)    /* the table as stored: thr[t] must be monotone in t, else T_k uses the prefix */
            if (down[j] ? thr[(size_t)j * K + t] > thr[(size_t)j * K + t + 1] : thr[(size_t)j * K + t] < thr[(size_t)j * K + t + 1]) { ++nonmono; break; }
        /* smallest F for this channel (search up to 30) */
        int fc = -1;
        for (int g = 0; g <= 30 && fc < 0; ++g) { int64_t lo, hi; if (theta_range(T_, (int64_t)1 << g, &lo, &hi)) fc = g; }
        if (fc < 0) fc = 99;
        if (fc > fmin_layer) fmin_layer = fc;
        int64_t lo, hi; int hits;
        if (theta_range(T_, S, &lo, &hi)) {
            /* the theta nearest the mean spacing inside the interval */
            int64_t est = (int64_t)((double)(T_[K] - T_[1]) / (K - 1) * S + 0.5);
            theta[j] = est < lo ? lo : est > hi ? hi : est;
            mem0[j] = mem0_for(T_, S, theta[j], &hits); assert(hits == K);
        } else {
            ++inexact_ch;
            int64_t est = (int64_t)((double)(T_[K] - T_[1]) / (K - 1) * S + 0.5);
            if (est < 1) est = 1;
            int best = -1; int64_t bt = est, bm = 0;
            for (int64_t th = est > 64 ? est - 64 : 1; th <= est + 64; ++th) {
                int64_t mm = mem0_for(T_, S, th, &hits);
                if (hits > best) { best = hits; bt = th; bm = mm; }
            }
            theta[j] = bt; mem0[j] = bm; hits = best;
        }
        if (hits < min_hits) min_hits = hits;
    }

    /* every neuron: table level, PASC-parameter level (floor formula), and the three-stage PASC-IF
     * recurrence on the real O[t] (fixed point); ranges of pseudo, corr, O, and the membrane */
    int64_t pse[2] = {INT64_MAX, INT64_MIN}, cor[2] = {INT64_MAX, INT64_MIN}, ov[2] = {INT64_MAX, INT64_MIN},
            mem[2] = {INT64_MAX, INT64_MIN}, thx = 0, m0r[2] = {INT64_MAX, INT64_MIN};
    uint64_t neurons = 0, mis_formula = 0, mis_pasc = 0, nonsilent = 0;
#define UPD(r, v) do { if ((v) < (r)[0]) (r)[0] = (v); if ((v) > (r)[1]) (r)[1] = (v); } while (0)
    for (int j = 0; j < m; ++j) { if (theta[j] > thx) thx = theta[j]; UPD(m0r, mem0[j]); }
    int64_t Sl[16];
    for (int p = 0; p < P; ++p) for (int j = 0; j < m; ++j) {
        memset(Sl, 0, sizeof Sl);
        const int8_t *w = &wt[(size_t)j * n]; const uint8_t *l = &lev[(size_t)p * n];
        for (int i = 0; i < n; ++i) Sl[l[i]] += w[i];
        int64_t V = 0, pseudo = 0, corr[16], O[16];
        for (int L = 1; L <= K; ++L) { V += (int64_t)L * Sl[L]; pseudo += Sl[L]; }
        int64_t c = 0;
        for (int t = 0; t < K; ++t) { if (t >= 1) c += Sl[t]; corr[t] = c; O[t] = pseudo - c; }   /* corr[t]: inputs with 1 <= level <= t */
        UPD(pse, pseudo);
        for (int t = 0; t < K; ++t) { UPD(cor, corr[t]); UPD(ov, O[t]); }
        int t = 0; while (t < K && (down[j] ? V > thr[(size_t)j * K + t] : V < thr[(size_t)j * K + t])) ++t;
        int want = t < K ? K - t : 0;
        int64_t d = down[j] ? -1 : 1;
        int64_t q = fdiv(d * V * S + mem0[j], theta[j]);
        int lf = q < 0 ? 0 : q > K ? K : (int)q;
        /* PASC-IF, PASCAL Alg. 1 with L(n-1) = L(n) = K, as in rtl/apex.sv apex_pasc */
        int64_t mm = mem0[j], s = 0;
        UPD(mem, mm);
        for (int u = 0; u < K; ++u) { mm += d * O[u] * S; UPD(mem, mm); if (mm >= theta[j]) { ++s; mm -= theta[j]; } UPD(mem, mm); }
        for (int u = 0; u < K - 1; ++u) {
            if (mm >= theta[j]) { ++s; mm -= theta[j]; } else if (mm < 0) { --s; mm += theta[j]; }
            UPD(mem, mm);
        }
        int lp = 0; for (int u = 0; u < K; ++u) if (s >= 1) { ++lp; --s; }
        ++neurons; nonsilent += want != 0;
        mis_formula += lf != want; mis_pasc += lp != want;
    }
    if (strcmp(argv[3], "-")) {
        FILE *o = fopen(argv[3], "w"); assert(o);
        fprintf(o, "%d %d\n", m, F);
        for (int j = 0; j < m; ++j) fprintf(o, "%" PRId64 " %" PRId64 " %d\n", theta[j], mem0[j], down[j]);
        fclose(o);
    }
    printf("{\"file\": \"%s\", \"n\": %d, \"m\": %d, \"P\": %d, \"K\": %d, \"F\": %d, \"F_needed\": %d, \"channels_nonmonotone_table\": %d, "
           "\"channels_inexact\": %d, \"min_levels_exact\": %d, \"neurons\": %" PRIu64 ", \"nonsilent_outputs\": %" PRIu64 ", "
           "\"mismatch_formula\": %" PRIu64 ", \"mismatch_pasc\": %" PRIu64 ", \"pseudo\": [%" PRId64 ", %" PRId64 "], \"corr\": [%" PRId64 ", %" PRId64 "], "
           "\"O\": [%" PRId64 ", %" PRId64 "], \"mem\": [%" PRId64 ", %" PRId64 "], \"theta_max\": %" PRId64 ", \"mem0\": [%" PRId64 ", %" PRId64 "]}\n",
           argv[1], n, m, P, K, F, fmin_layer, nonmono, inexact_ch, min_hits, neurons, nonsilent, mis_formula, mis_pasc,
           pse[0], pse[1], cor[0], cor[1], ov[0], ov[1], mem[0], mem[1], thx, m0r[0], m0r[1]);
    return 0;
}

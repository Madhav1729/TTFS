#ifndef BRIDGE_H
#define BRIDGE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* C interface of the Verilator models (SparTen cluster and TTFS tile) used by harness.c. */
#ifndef UNITS_CFG
#define UNITS_CFG 32
#endif
#ifndef LANES_CFG
#define LANES_CFG 64
#endif
#ifndef OB_CFG
#define OB_CFG 32
#endif
#ifndef CH_CFG
#define CH_CFG 256
#endif
#ifndef EDEPTH_CFG
#define EDEPTH_CFG 32
#endif
#ifndef SPTW_CFG
#define SPTW_CFG 4
#endif
#ifndef SPTA_CFG
#define SPTA_CFG 4
#endif
#ifndef WS_CFG
#define WS_CFG 16
#endif
#ifndef STEPS_CFG
#define STEPS_CFG 8
#endif
#ifndef TBITS_CFG
#define TBITS_CFG 3
#endif
#ifndef NPORT_CFG
#define NPORT_CFG 4
#endif
enum { SPT, TTF };
enum {
    UNITS = UNITS_CFG,
    NF = 2 * UNITS_CFG,
    LANES = LANES_CFG,
    BLOCK = 128,
    OB = OB_CFG,
    CH = CH_CFG,
    EDEPTH = EDEPTH_CFG,
    NPORT = NPORT_CFG,
    SPT_W = SPTW_CFG,
    SPT_A = SPTA_CFG,
    WS = WS_CFG,
    PUSHW = 5,
    TT_W = 4,
    STEPS = STEPS_CFG,
    TBITS = TBITS_CFG,
    MASK_WORDS = (BLOCK + 63) / 64
};
typedef uint64_t lanes_t; /* one bit per unit / lane (<= 64) */
typedef struct {
    uint64_t w[MASK_WORDS];
} block_t; /* one bit per chunk position */

void r5_open(const char *directory, int trace);
void r5_close(void);
void r5_tick(int engine);
void r5_reset(int engine, int value);
void r5_idle(int engine); /* drive every one-shot pin low */

/* SparTen */
void spf_mask_write(lanes_t enables, int buffer, int filter, const block_t *masks);
void spf_value_write(lanes_t enables, int buffer, int filter, int address, const int *values);
void spf_input_write(int buffer, const block_t *mask, const int *values);
void spf_start(int buffer, int in_buffer, int first, int bank, int slot, const int *active2,
               const int *dest2);
void spf_collect(int bank, int pixels, const uint8_t *active_nf);
void spf_threshold(int fil, int t, int32_t value, int down);
int spf_busy(void);
int spf_holding(void);
int spf_net_busy(void);
int spf_col_busy(void);
int spf_out(int *pixel, uint8_t *mask_nf, uint8_t *packed_nf, int32_t *sums_nf); /* 1 if a pixel came out */
lanes_t spf_reads(void);
lanes_t spf_macs(void);
lanes_t spf_takes(void);
int spf_moves(void);

/* TTFS tile */
void tt_active(lanes_t active);
void tt_wb_write(int buffer, int address, lanes_t mask, const int *packed);
void tt_ev_push(int n, const int *index, const int *tick); /* up to PUSHW events */
int tt_ev_count(void);
int tt_pass_idle(void);
int tt_ramp_busy(void);
void tt_pass_start(int set, int wbuf, int bank, int slot, int first, int len);
void tt_pass_end(void);
void tt_threshold(int timestep, const int32_t *data, lanes_t down);
void tt_cmp_load(int bank, int slot);
void tt_compare(int bank, int slot, int timestep);
lanes_t tt_spikes(void);
int tt_out(int *count, int *indices); /* 1 if a compacted spike list came out */
int32_t tt_result(int lane);
lanes_t tt_adds(void);
int tt_issued(void);
int tt_col_reads(void);
#ifdef __cplusplus
}
#endif
#endif

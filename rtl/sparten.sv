// SparTen cluster (Gondimalla et al., MICRO'19) at W4/A4.

// Priority encoder over the AND of two SparseMaps.
module spt_priority #(parameter integer BLOCK = 16, IDX = $clog2(BLOCK)) (
    input  logic [BLOCK-1:0] bits,
    output logic found,
    output logic [IDX-1:0] position
);
    always_comb begin                               // lowest set position first
        found = 0;
        position = '0;
        for (integer i = BLOCK-1; i >= 0; i = i-1)
            if (bits[i]) begin
                found = 1;
                position = IDX'(i);
            end
    end
endmodule

// Input chunk buffer (SRAM-like).
module spt_input_buffer #(parameter integer BLOCK = 16, A = 9) (
    input  logic clk, write, wbuf,
    input  logic [BLOCK-1:0] mask_data,
    input  logic [BLOCK*A-1:0] value_data,
    input  logic mask_buf, value_buf,
    output logic [BLOCK-1:0] mask,
    output logic [BLOCK*A-1:0] values
);
    logic [BLOCK-1:0] masks [0:1];
    logic [BLOCK*A-1:0] chunk [0:1];
    always_ff @(posedge clk) if (write) begin
        masks[wbuf] <= mask_data;
        chunk[wbuf] <= value_data;
    end
    assign mask = masks[mask_buf];
    assign values = chunk[value_buf];
endmodule

// Filter chunk buffer (SRAM-like).
module spf_filter_buffer #(parameter integer BLOCK = 128, W = 8, IDX = $clog2(BLOCK)) (
    input  logic clk,
    input  logic mask_write, value_write, wbuf, wfil,
    input  logic [BLOCK-1:0] mask_data,
    input  logic [IDX-1:0] value_addr,
    input  logic signed [W-1:0] value_data,
    input  logic rbuf,
    input  logic sel_fil, rank_fil, val_fil,
    input  logic [IDX-1:0] val_addr,
    output logic [BLOCK-1:0] sel_mask, rank_mask,
    output logic signed [W-1:0] value
);
    logic [BLOCK-1:0] masks [0:3];
    logic signed [W-1:0] values [0:3][0:BLOCK-1];
    always_ff @(posedge clk) begin
        if (mask_write) masks[{wbuf, wfil}] <= mask_data;
        if (value_write) values[{wbuf, wfil}][value_addr] <= value_data;
    end
    assign sel_mask = masks[{rbuf, sel_fil}];
    assign rank_mask = masks[{rbuf, rank_fil}];
    assign value = values[{rbuf, val_fil}][val_addr];
endmodule

// Packed activation read.
module spf_act_read #(parameter integer BLOCK = 128, A = 8, IDX = $clog2(BLOCK)) (
    input  logic [BLOCK*A-1:0] values,
    input  logic [IDX-1:0] rank,
    output logic [A-1:0] value
);
    assign value = values[int'(rank)*A +: A];
endmodule

// Output-sum buffer of a home unit (SRAM-like).
module spf_sum_buffer #(parameter integer OB = 32, ACC = 32, SLOT = $clog2(OB)) (
    input  logic clk,
    input  logic write, bank, fil,
    input  logic [SLOT-1:0] addr,
    input  logic [ACC-1:0] wdata,
    output logic [ACC-1:0] rdata,
    input  logic obank,
    input  logic [SLOT-1:0] oaddr,
    output logic [2*ACC-1:0] odata
);
    logic [ACC-1:0] mem [0:4*OB-1];
    always_ff @(posedge clk) if (write) mem[{bank, addr, fil}] <= wdata;
    assign rdata = mem[{bank, addr, fil}];
    assign odata = {mem[{obank, oaddr, 1'b1}], mem[{obank, oaddr, 1'b0}]};
endmodule

// Per-group popcounts of a SparseMap.
module spf_prefix_groups #(parameter integer BLOCK = 128, GW = 32, IDX = $clog2(BLOCK),
                           NG = BLOCK / GW, CW = $clog2(GW) + 1) (
    input  logic [BLOCK-1:0] mask,
    input  logic [IDX-1:0] position,
    output logic [NG*CW-1:0] counts
);
    always_comb
        for (int g = 0; g < NG; g++) begin
            logic [CW-1:0] c;
            c = '0;
            for (int i = 0; i < GW; i++)
                if (IDX'(g*GW + i) < position) c = c + CW'(mask[g*GW + i]);
            counts[g*CW +: CW] = c;
        end
endmodule

// Prefix sum over the group counts.
module spf_prefix_sum #(parameter integer BLOCK = 128, GW = 32, IDX = $clog2(BLOCK),
                        NG = BLOCK / GW, CW = $clog2(GW) + 1) (
    input  logic [NG*CW-1:0] counts,
    output logic [IDX-1:0] rank
);
    always_comb begin
        rank = '0;
        for (int g = 0; g < NG; g++) rank = rank + IDX'(counts[g*CW +: CW]);
    end
endmodule

// Compute unit: inner join and multiply-accumulate.
module spf_unit #(parameter integer BLOCK = 128, W = 8, A = 8, ACC = 32, OB = 32, NF = 64,
                  IDX = $clog2(BLOCK), SLOT = $clog2(OB), FID = $clog2(NF)) (
    input  logic clk, reset,
    input  logic mask_write, value_write, wbuf, wfil,
    input  logic [BLOCK-1:0] mask_data,
    input  logic [IDX-1:0] value_addr,
    input  logic signed [W-1:0] value_data,
    input  logic start, buf_sel, first, bank,
    input  logic [SLOT-1:0] slot,
    input  logic [1:0] active,
    input  logic [2*FID-1:0] dest,                  // output-sum index of each held filter chunk
    input  logic [BLOCK-1:0] amask,                 // current input chunk (stable while busy)
    input  logic [BLOCK*A-1:0] avalues,
    input  logic take,                              // the network took this unit's partial sums
    output logic busy,
    output logic out_valid,                         // partial sums waiting for the network
    output logic [2*ACC-1:0] out_partial,
    output logic [2*FID-1:0] out_dest,
    output logic [1:0] out_active,
    output logic [SLOT-1:0] out_slot,
    output logic out_first, out_bank,
    output logic read_weight, do_mac
);
    logic cur_buf, cur_fil, second, job_first, job_bank, emit_wait;
    logic [SLOT-1:0] job_slot;
    logic [1:0] job_active;
    logic [2*FID-1:0] job_dest;
    logic [BLOCK-1:0] remaining, wmask, rank_mask;      // wmask: filter 0's SparseMap (read at start)
    logic signed [W-1:0] wvalue;
    logic s1_valid, s1_fil, s2_valid, s2_fil, s3_valid, s3_fil, s4_valid, s4_fil;
    logic [IDX-1:0] s1_pos, wrank, arank, wrank_q, arank_q;
    spf_filter_buffer #(.BLOCK(BLOCK), .W(W)) fbuf (
        .clk(clk), .mask_write(mask_write), .value_write(value_write), .wbuf(wbuf), .wfil(wfil),
        .mask_data(mask_data), .value_addr(value_addr), .value_data(value_data),
        .rbuf(start ? buf_sel : cur_buf), .sel_fil(1'b0), .rank_fil(start ? 1'b1 : s1_fil),
        .val_fil(s2_fil), .val_addr(wrank_q), .sel_mask(wmask), .rank_mask(rank_mask), .value(wvalue));
    // stage 0: AND the SparseMaps once per job, then a priority encoder walks the matches
    logic [BLOCK-1:0] live1;
    logic found;
    logic [IDX-1:0] position;
    spt_priority #(.BLOCK(BLOCK)) prio (.bits(remaining), .found(found), .position(position));
    wire select = busy && !emit_wait && found;
    // stages 1a/1b: ranks (one prefix sum per tensor: group counts, then their sum)
    localparam integer PG = BLOCK / 32, PCW = 6;
    logic [PG*PCW-1:0] wcounts, acounts, wcounts_q, acounts_q;
    logic s1b_valid, s1b_fil;
    spf_prefix_groups #(.BLOCK(BLOCK)) wpre (.mask(rank_mask), .position(s1_pos), .counts(wcounts));
    spf_prefix_groups #(.BLOCK(BLOCK)) apre (.mask(amask), .position(s1_pos), .counts(acounts));
    spf_prefix_sum #(.BLOCK(BLOCK)) wsum (.counts(wcounts_q), .rank(wrank));
    spf_prefix_sum #(.BLOCK(BLOCK)) asum (.counts(acounts_q), .rank(arank));
    // stage 2: packed-value reads (filter value above; activation from the input chunk copy)
    logic [A-1:0] avalue;
    spf_act_read #(.BLOCK(BLOCK), .A(A)) aread (.values(avalues), .rank(arank_q), .value(avalue));
    // stage 3: multiply; stage 4: accumulate into the job's partial sums
    logic signed [W-1:0] weight_reg;
    logic signed [A:0] act_reg;
    logic signed [W+A:0] product;
    logic signed [ACC-1:0] partial [0:1];
    assign read_weight = s2_valid;
    assign do_mac = s4_valid;
    wire pipe_empty = !s1_valid && !s1b_valid && !s2_valid && !s3_valid && !s4_valid;
    wire next_filter = busy && !emit_wait && !found && !second && job_active[1];
    wire job_done = busy && !emit_wait && !found && (second || !job_active[1]) && pipe_empty;
    wire can_emit = !out_valid || take;

    always_ff @(posedge clk) begin
        if (reset) begin
            busy <= 0; emit_wait <= 0; out_valid <= 0;
            cur_buf <= 0; cur_fil <= 0; second <= 0;
            remaining <= '0; live1 <= '0; job_slot <= '0; job_active <= '0;
            job_dest <= '0; job_first <= 0; job_bank <= 0;
            s1_valid <= 0; s1b_valid <= 0; s2_valid <= 0; s3_valid <= 0; s4_valid <= 0;
            s1_fil <= 0; s1b_fil <= 0; s2_fil <= 0; s3_fil <= 0; s4_fil <= 0;
            wcounts_q <= '0; acounts_q <= '0;
            s1_pos <= '0; wrank_q <= '0; arank_q <= '0;
            weight_reg <= '0; act_reg <= '0; product <= '0;
            partial[0] <= '0; partial[1] <= '0;
            out_partial <= '0; out_dest <= '0; out_active <= '0; out_slot <= '0; out_first <= 0; out_bank <= 0;
        end else begin
            if (take) out_valid <= 0;
            if (start) begin
                cur_buf <= buf_sel; cur_fil <= 0; second <= 0;
                job_slot <= slot; job_active <= active; job_dest <= dest; job_first <= first; job_bank <= bank;
                remaining <= active[0] ? amask & wmask : '0;
                live1 <= active[1] ? amask & rank_mask : '0;
                partial[0] <= '0; partial[1] <= '0;
                busy <= |active;
            end else begin
                if (select) remaining[position] <= 0;
                if (next_filter) begin
                    cur_fil <= 1; second <= 1;
                    remaining <= live1;
                end
                if ((job_done || emit_wait) && can_emit) begin   // hand the pair to the network
                    out_valid <= 1; emit_wait <= 0; busy <= 0;
                    out_partial <= {partial[1], partial[0]};
                    out_dest <= job_dest; out_active <= job_active; out_slot <= job_slot;
                    out_first <= job_first; out_bank <= job_bank;
                end else if (job_done) emit_wait <= 1;
            end
            s1_valid <= select;
            s1_fil <= cur_fil;
            s1_pos <= position;
            s1b_valid <= s1_valid; s1b_fil <= s1_fil; wcounts_q <= wcounts; acounts_q <= acounts;
            s2_valid <= s1b_valid; s2_fil <= s1b_fil; wrank_q <= wrank; arank_q <= arank;
            s3_valid <= s2_valid; s3_fil <= s2_fil;
            if (s2_valid) begin weight_reg <= wvalue; act_reg <= $signed({1'b0, avalue}); end
            s4_valid <= s3_valid; s4_fil <= s3_fil;
            if (s3_valid) product <= weight_reg * act_reg;
            if (s4_valid) partial[s4_fil] <= partial[s4_fil] + ACC'(product);
        end
    end
    // synthesis translate_off
    always_ff @(posedge clk) if (!reset) begin
        assert (!(start && busy)) else $fatal(1, "start while unit busy");
        assert (!(start && !pipe_empty)) else $fatal(1, "start before the pipeline drained");
        assert (!(start && emit_wait)) else $fatal(1, "start while partial sums wait");
        assert (!((mask_write || value_write) && (busy || !pipe_empty) && wbuf == cur_buf))
            else $fatal(1, "write into the buffer being computed");
    end
    // synthesis translate_on
endmodule

// Permutation network for GB-H partial sums.
module spf_network #(parameter integer UNITS = 32, ACC = 32, OB = 32, NPORT = 4, NF = 64,
                     SLOT = $clog2(OB), FID = $clog2(NF), SRC = $clog2(2*UNITS)) (
    input  logic clk, reset,
    input  logic [UNITS-1:0] in_valid,
    input  logic [UNITS*2*ACC-1:0] in_partial,
    input  logic [UNITS*2*FID-1:0] in_dest,
    input  logic [UNITS*2-1:0] in_active,
    input  logic [UNITS*SLOT-1:0] in_slot,
    input  logic [UNITS-1:0] in_first, in_bank,
    output logic [UNITS-1:0] take,
    output logic busy,
    // read-modify-write ports of the home units' output-sum buffers
    output logic [UNITS-1:0] sum_write, sum_bank, sum_fil,
    output logic [UNITS*SLOT-1:0] sum_addr,
    output logic [UNITS*ACC-1:0] sum_data,
    input  logic [UNITS*ACC-1:0] sum_old,
    output logic [NPORT-1:0] moves
);
    logic [2*UNITS-1:0] moved, pending;
    always_comb
        for (int s = 0; s < 2*UNITS; s++)
            pending[s] = in_valid[s/2] && in_active[s] && !moved[s];
    // per port: the lowest pending source whose destination home belongs to the port
    logic [NPORT-1:0] g_valid;
    logic [SRC-1:0] g_src [0:NPORT-1];
    always_comb
        for (int p = 0; p < NPORT; p++) begin
            g_valid[p] = 0;
            g_src[p] = '0;
            for (int s = 2*UNITS-1; s >= 0; s--)
                if (pending[s] && int'(in_dest[s*FID+1 +: FID-1]) % NPORT == p) begin
                    g_valid[p] = 1;
                    g_src[p] = SRC'(s);
                end
        end
    assign moves = g_valid;
    // a unit's pair is taken once nothing of it is left after this cycle's moves
    always_comb
        for (int u = 0; u < UNITS; u++) begin
            logic left0, left1;
            left0 = pending[2*u];
            left1 = pending[2*u+1];
            for (int p = 0; p < NPORT; p++) begin
                if (g_valid[p] && int'(g_src[p]) == 2*u) left0 = 0;
                if (g_valid[p] && int'(g_src[p]) == 2*u+1) left1 = 0;
            end
            take[u] = in_valid[u] && !left0 && !left1;
        end
    // stage 2: registered moves
    logic [NPORT-1:0] m_valid, m_first, m_bank;
    logic [FID-1:0] m_dest [0:NPORT-1];
    logic [SLOT-1:0] m_slot [0:NPORT-1];
    logic [ACC-1:0] m_val [0:NPORT-1];
    always_ff @(posedge clk) begin
        if (reset) begin
            moved <= '0; m_valid <= '0; m_first <= '0; m_bank <= '0;
            for (int p = 0; p < NPORT; p++) begin m_dest[p] <= '0; m_slot[p] <= '0; m_val[p] <= '0; end
        end else begin
            for (int p = 0; p < NPORT; p++) begin
                m_valid[p] <= g_valid[p];
                if (g_valid[p]) begin
                    m_dest[p] <= in_dest[int'(g_src[p])*FID +: FID];
                    m_slot[p] <= in_slot[(int'(g_src[p]) / 2)*SLOT +: SLOT];
                    m_val[p] <= in_partial[int'(g_src[p])*ACC +: ACC];
                    m_first[p] <= in_first[int'(g_src[p]) / 2];
                    m_bank[p] <= in_bank[int'(g_src[p]) / 2];
                end
            end
            for (int s = 0; s < 2*UNITS; s++) begin
                if (take[s/2]) moved[s] <= 0;
                else for (int p = 0; p < NPORT; p++)
                    if (g_valid[p] && int'(g_src[p]) == s) moved[s] <= 1;
            end
        end
    end
    assign busy = |pending || |m_valid;
    // addresses (no dependence on the old sums) ...
    always_comb begin
        sum_write = '0; sum_bank = '0; sum_fil = '0; sum_addr = '0;
        for (int p = 0; p < NPORT; p++)
            if (m_valid[p]) begin
                sum_write[int'(m_dest[p]) / 2] = 1;
                sum_bank[int'(m_dest[p]) / 2] = m_bank[p];
                sum_fil[int'(m_dest[p]) / 2] = m_dest[p][0];
                sum_addr[(int'(m_dest[p]) / 2)*SLOT +: SLOT] = m_slot[p];
            end
    end
    // ... and the new sums
    always_comb begin
        sum_data = '0;
        for (int p = 0; p < NPORT; p++)
            if (m_valid[p])
                sum_data[(int'(m_dest[p]) / 2)*ACC +: ACC] =
                    m_first[p] ? m_val[p] : sum_old[(int'(m_dest[p]) / 2)*ACC +: ACC] + m_val[p];
    end
endmodule

// Output collector: threshold compare and compaction.
module spf_collector #(parameter integer UNITS = 32, NF = 64, ACC = 32, OB = 32, T = 8, A = 8,
                       SLOT = $clog2(OB), TIME = $clog2(T), FID = $clog2(NF)) (
    input  logic clk, reset,
    input  logic start, bank,
    input  logic [SLOT:0] pixels,
    input  logic [NF-1:0] active,
    input  logic [UNITS*2*ACC-1:0] sums,            // the selected pixel's row of every home buffer
    output logic obank,
    output logic [SLOT-1:0] oaddr,
    input  logic th_write,
    input  logic [FID-1:0] th_fil,
    input  logic [TIME-1:0] th_t,
    input  logic [ACC-1:0] th_data,
    input  logic th_down,
    output logic busy,
    output logic out_valid,
    output logic [NF-1:0] out_mask,
    output logic [NF*A-1:0] out_packed,
    output logic [SLOT-1:0] out_pixel,
    output logic [NF*ACC-1:0] out_sums
);
    logic signed [ACC-1:0] table_ [0:NF-1][0:T-1];
    logic [NF-1:0] down;
    always_ff @(posedge clk) if (th_write) begin
        table_[th_fil][th_t] <= $signed(th_data);
        down[th_fil] <= th_down;
    end
    typedef enum logic [1:0] { IDLE, LOAD, CMP, PACK } state_t;
    state_t state;
    logic [SLOT:0] left;
    logic [SLOT-1:0] px;
    logic [TIME:0] t;
    logic signed [ACC-1:0] v [0:NF-1];
    logic [A-1:0] level [0:NF-1];
    logic [NF-1:0] seen;
    assign busy = state != IDLE;
    assign oaddr = px;
    // compaction network (inverted prefix sum + log-stage left shifts)
    logic [NF*A-1:0] stage [0:FID];
    logic [NF*FID-1:0] shift [0:FID];
    logic [NF-1:0] live [0:FID];
    always_comb begin
        logic [FID:0] zeros;
        zeros = '0;
        for (int i = 0; i < NF; i++) begin
            stage[0][i*A +: A] = level[i];
            shift[0][i*FID +: FID] = FID'(zeros);
            live[0][i] = level[i] != '0;
            zeros = zeros + (FID+1)'(level[i] == '0);
        end
        for (int k = 0; k < FID; k++) begin
            stage[k+1] = '0; shift[k+1] = '0; live[k+1] = '0;
            for (int i = 0; i < NF; i++)
                if (live[k][i]) begin
                    if (shift[k][i*FID + k] && i >= (1 << k)) begin
                        stage[k+1][(i - (1 << k))*A +: A] = stage[k][i*A +: A];
                        shift[k+1][(i - (1 << k))*FID +: FID] = shift[k][i*FID +: FID];
                        live[k+1][i - (1 << k)] = 1;
                    end else begin
                        stage[k+1][i*A +: A] = stage[k][i*A +: A];
                        shift[k+1][i*FID +: FID] = shift[k][i*FID +: FID];
                        live[k+1][i] = 1;
                    end
                end
        end
    end
    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE; left <= '0; px <= '0; t <= '0; out_valid <= 0; obank <= 0;
            out_mask <= '0; out_packed <= '0; out_pixel <= '0; seen <= '0; out_sums <= '0;
            for (int f = 0; f < NF; f++) begin v[f] <= '0; level[f] <= '0; end
        end else begin
            out_valid <= 0;
            case (state)
                IDLE: if (start) begin
                    obank <= bank; px <= '0; left <= pixels;
                    if (pixels != '0) state <= LOAD;
                end
                LOAD: begin                          // pixel px's sums are on `sums` this cycle
                    for (int f = 0; f < NF; f++) begin
                        v[f] <= $signed(sums[f*ACC +: ACC]);
                        level[f] <= '0;
                    end
                    seen <= '0; t <= '0; state <= CMP;
                end
                CMP: begin
                    for (int f = 0; f < NF; f++)
                        if (active[f] && !seen[f] &&
                            (down[f] ? v[f] <= table_[f][t[TIME-1:0]] : v[f] >= table_[f][t[TIME-1:0]])) begin
                            seen[f] <= 1;
                            level[f] <= A'(T) - A'(t);
                        end
                    if (t == (TIME+1)'(T-1)) state <= PACK;
                    t <= t + 1'b1;
                end
                PACK: begin
                    out_valid <= 1;
                    out_pixel <= px;
                    out_packed <= stage[FID];
                    for (int f = 0; f < NF; f++) begin
                        out_mask[f] <= level[f] != '0;
                        out_sums[f*ACC +: ACC] <= v[f];
                    end
                    px <= px + 1'b1;
                    left <= left - 1'b1;
                    state <= left == (SLOT+1)'(1) ? IDLE : LOAD;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule

// Cluster: 32 units, network, output sums, collector.
module sparten_full #(
    parameter integer UNITS = 32, BLOCK = 128, W = 8, A = 8, ACC = 32, OB = 32, NPORT = 4, T = 8,
    parameter integer NF = 2*UNITS, IDX = $clog2(BLOCK), SLOT = $clog2(OB), FID = $clog2(NF),
    parameter integer TIME = $clog2(T)
) (
    input  logic clk, reset,
    // filter chunk loads (per unit, collocated pair)
    input  logic [UNITS-1:0] mask_write, value_write,
    input  logic wbuf, wfil,
    input  logic [UNITS*BLOCK-1:0] mask_data,
    input  logic [IDX-1:0] value_addr,
    input  logic [UNITS*W-1:0] value_data,
    // broadcast input chunk (double-buffered)
    input  logic in_write, in_buf,
    input  logic [BLOCK-1:0] in_mask,
    input  logic [BLOCK*A-1:0] in_data,
    // job start (all units together: same pixel, same chunk)
    input  logic start, buf_sel, in_sel, first, bank,
    input  logic [SLOT-1:0] slot,
    input  logic [UNITS*2-1:0] active,
    input  logic [UNITS*2*FID-1:0] dest,
    // output collector
    input  logic collect, collect_bank,
    input  logic [SLOT:0] collect_pixels,
    input  logic [NF-1:0] collect_active,
    input  logic th_write,
    input  logic [FID-1:0] th_fil,
    input  logic [TIME-1:0] th_t,
    input  logic [ACC-1:0] th_data,
    input  logic th_down,
    output logic busy, holding, net_busy, col_busy,
    output logic out_valid,
    output logic [NF-1:0] out_mask,
    output logic [NF*A-1:0] out_packed,
    output logic [SLOT-1:0] out_pixel,
    output logic [NF*ACC-1:0] out_sums,
    output wire [UNITS-1:0] read_weight, do_mac, unit_done,
    output wire [NPORT-1:0] moves
);
    logic cur_in;
    wire [UNITS-1:0] unit_busy, unit_out, take;
    wire [BLOCK-1:0] cur_mask;
    wire [BLOCK*A-1:0] cur_values;
    wire [UNITS*2*ACC-1:0] partials;
    wire [UNITS*2*FID-1:0] dests;
    wire [UNITS*2-1:0] actives;
    wire [UNITS*SLOT-1:0] slots;
    wire [UNITS-1:0] firsts, banks;
    always_ff @(posedge clk) begin
        if (reset) cur_in <= 0;
        else if (start) cur_in <= in_sel;
    end
    spt_input_buffer #(.BLOCK(BLOCK), .A(A)) ibuf (
        .clk(clk), .write(in_write), .wbuf(in_buf), .mask_data(in_mask), .value_data(in_data),
        .mask_buf(start ? in_sel : cur_in), .value_buf(cur_in), .mask(cur_mask), .values(cur_values));
    // Units are free for the next job once their partial sums are latched for the network (the
    // network moves them under the next job's computation, as in the paper).
    assign busy = |unit_busy;
    assign holding = |unit_out;
    assign unit_done = take;
    for (genvar u = 0; u < UNITS; u = u+1) begin : units
        spf_unit #(.BLOCK(BLOCK), .W(W), .A(A), .ACC(ACC), .OB(OB), .NF(NF)) pe (
            .clk(clk), .reset(reset),
            .mask_write(mask_write[u]), .value_write(value_write[u]), .wbuf(wbuf), .wfil(wfil),
            .mask_data(mask_data[u*BLOCK +: BLOCK]), .value_addr(value_addr),
            .value_data(value_data[u*W +: W]), .start(start), .buf_sel(buf_sel), .first(first),
            .bank(bank), .slot(slot), .active(active[2*u +: 2]), .dest(dest[u*2*FID +: 2*FID]),
            .amask(cur_mask), .avalues(cur_values), .take(take[u]),
            .busy(unit_busy[u]), .out_valid(unit_out[u]), .out_partial(partials[u*2*ACC +: 2*ACC]),
            .out_dest(dests[u*2*FID +: 2*FID]), .out_active(actives[2*u +: 2]),
            .out_slot(slots[u*SLOT +: SLOT]), .out_first(firsts[u]), .out_bank(banks[u]),
            .read_weight(read_weight[u]), .do_mac(do_mac[u]));
    end
    // permutation network and the home units' output-sum buffers
    wire [UNITS-1:0] s_write, s_bank, s_fil;
    wire [UNITS*SLOT-1:0] s_addr;
    wire [UNITS*ACC-1:0] s_data, s_old;
    wire [UNITS*2*ACC-1:0] col_rows;
    wire col_bank;
    wire [SLOT-1:0] col_addr;
    spf_network #(.UNITS(UNITS), .ACC(ACC), .OB(OB), .NPORT(NPORT), .NF(NF)) net (
        .clk(clk), .reset(reset), .in_valid(unit_out), .in_partial(partials), .in_dest(dests),
        .in_active(actives), .in_slot(slots), .in_first(firsts), .in_bank(banks), .take(take),
        .busy(net_busy), .sum_write(s_write), .sum_bank(s_bank), .sum_fil(s_fil), .sum_addr(s_addr),
        .sum_data(s_data), .sum_old(s_old), .moves(moves));
    for (genvar h = 0; h < UNITS; h = h+1) begin : homes
        spf_sum_buffer #(.OB(OB), .ACC(ACC)) outsums (
            .clk(clk), .write(s_write[h]), .bank(s_bank[h]), .fil(s_fil[h]),
            .addr(s_addr[h*SLOT +: SLOT]), .wdata(s_data[h*ACC +: ACC]), .rdata(s_old[h*ACC +: ACC]),
            .obank(col_bank), .oaddr(col_addr), .odata(col_rows[h*2*ACC +: 2*ACC]));
    end
    spf_collector #(.UNITS(UNITS), .NF(NF), .ACC(ACC), .OB(OB), .T(T), .A(A)) collector (
        .clk(clk), .reset(reset), .start(collect), .bank(collect_bank), .pixels(collect_pixels),
        .active(collect_active), .sums(col_rows), .obank(col_bank), .oaddr(col_addr),
        .th_write(th_write), .th_fil(th_fil), .th_t(th_t), .th_data(th_data), .th_down(th_down),
        .busy(col_busy), .out_valid(out_valid), .out_mask(out_mask), .out_packed(out_packed),
        .out_pixel(out_pixel), .out_sums(out_sums));
    // synthesis translate_off
    always_ff @(posedge clk) if (!reset) begin
        assert (!(in_write && busy && in_buf == cur_in)) else $fatal(1, "input write into active buffer");
        assert (!(collect && col_busy)) else $fatal(1, "collector busy");
    end
    // synthesis translate_on
endmodule

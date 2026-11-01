/* verilator lint_off DECLFILENAME */
// Eyeriss v1 PE (JSSC 2017, Fig. 12), INT4: I/O FIFOs, filter / ifmap / psum spads, 3-stage MAC,
// zero-ifmap gating (gates, does not skip). One pass: load p*q*S weights, then cfg_rows ifmap rows.

module eyr_fifo #(parameter int WD = 8, D = 2, CW = $clog2(D + 1)) (
    input  logic clk, reset,
    input  logic in_valid,
    output logic in_ready,
    input  logic [WD-1:0] in_data,
    output logic out_valid,
    input  logic out_ready,
    output logic [WD-1:0] out_data,
    output logic [CW-1:0] count
);
    logic [WD-1:0] mem [0:D-1];
    logic [$clog2(D)-1:0] rp, wp;
    wire push = in_valid && in_ready;
    wire pop = out_valid && out_ready;
    assign in_ready = count != CW'(D);
    assign out_valid = count != '0;
    assign out_data = mem[rp];
    always_ff @(posedge clk) begin
        if (reset) begin
            rp <= '0; wp <= '0; count <= '0;
        end else begin
            if (push) begin
                mem[wp] <= in_data;
                wp <= wp == $clog2(D)'(D - 1) ? '0 : wp + 1'b1;
            end
            if (pop) rp <= rp == $clog2(D)'(D - 1) ? '0 : rp + 1'b1;
            count <= count + CW'(push) - CW'(pop);
        end
    end
endmodule

// Filter spad: single-port-per-direction SRAM, synchronous read with read enable (gated on zeros).
module eyr_fspad #(parameter int W = 4, DEPTH = 224, AW = $clog2(DEPTH)) (
    input  logic clk,
    input  logic we,
    input  logic [AW-1:0] waddr,
    input  logic [W-1:0] wdata,
    input  logic re,
    input  logic [AW-1:0] raddr,
    output logic [W-1:0] rdata
);
    logic [W-1:0] mem [0:DEPTH-1];
    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        if (re) rdata <= mem[raddr];
    end
endmodule

module eyeriss_pe #(
    parameter int W = 4, A = 4, PSW = 20,
    parameter int FS = 224, IS = 12, PS = 24, FIFO_D = 2, OFIFO_D = 4,
    parameter int FAW = $clog2(FS), IAW = $clog2(IS)
) (
    input  logic clk, reset,
    // configuration (scan chain in the chip), stable during a pass
    input  logic [4:0] cfg_p,           // filters, 1..24
    input  logic [2:0] cfg_q,           // channels, 1..4 (q*S <= 12, p*q*S <= 224)
    input  logic [3:0] cfg_s,           // filter row length S, 1..12
    input  logic [3:0] cfg_u,           // stride U, 1..S
    input  logic [7:0] cfg_f,           // outputs per row (F)
    input  logic [7:0] cfg_rows,        // ifmap rows per pass (n images)
    input  logic cfg_pin,               // accumulate input psums
    input  logic start,                 // begin a pass (load filters, then compute)
    output logic busy,
    // filter GIN: 4 weights per beat
    input  logic f_valid,
    output logic f_ready,
    input  logic [4*W-1:0] f_data,
    // ifmap GIN: one value per beat
    input  logic i_valid,
    output logic i_ready,
    input  logic [A-1:0] i_data,
    // input psums (local network or psum GIN), 4 per beat
    input  logic pi_valid,
    output logic pi_ready,
    input  logic [4*PSW-1:0] pi_data,
    // output psums (local network or GON), 4 per beat
    output logic po_valid,
    input  logic po_ready,
    output logic [4*PSW-1:0] po_data
);
    // ---- I/O FIFOs ------------------------------------------------------------------------------
    logic fq_valid, fq_pop, iq_valid, iq_pop, pq_valid, pq_pop, oq_push, oq_ready;
    logic [4*W-1:0] fq_data;
    logic [A-1:0] iq_data;
    logic [4*PSW-1:0] pq_data, oq_data;
    logic [$clog2(OFIFO_D + 1)-1:0] oq_count;
    logic [$clog2(FIFO_D + 1)-1:0] unused_fc, unused_ic, unused_pc;
    eyr_fifo #(.WD(4*W), .D(FIFO_D)) fifo_f (.clk, .reset, .in_valid(f_valid), .in_ready(f_ready), .in_data(f_data),
        .out_valid(fq_valid), .out_ready(fq_pop), .out_data(fq_data), .count(unused_fc));
    eyr_fifo #(.WD(A), .D(FIFO_D)) fifo_i (.clk, .reset, .in_valid(i_valid), .in_ready(i_ready), .in_data(i_data),
        .out_valid(iq_valid), .out_ready(iq_pop), .out_data(iq_data), .count(unused_ic));
    eyr_fifo #(.WD(4*PSW), .D(FIFO_D)) fifo_pi (.clk, .reset, .in_valid(pi_valid), .in_ready(pi_ready), .in_data(pi_data),
        .out_valid(pq_valid), .out_ready(pq_pop), .out_data(pq_data), .count(unused_pc));
    eyr_fifo #(.WD(4*PSW), .D(OFIFO_D)) fifo_po (.clk, .reset, .in_valid(oq_push), .in_ready(oq_ready), .in_data(oq_data),
        .out_valid(po_valid), .out_ready(po_ready), .out_data(po_data), .count(oq_count));

    // ---- spads ----------------------------------------------------------------------------------
    logic fs_we, fs_re;
    logic [FAW-1:0] fs_waddr, fs_raddr;
    logic [W-1:0] fs_wdata, fs_rdata;
    eyr_fspad #(.W(W), .DEPTH(FS)) fspad (.clk, .we(fs_we), .waddr(fs_waddr), .wdata(fs_wdata), .re(fs_re),
        .raddr(fs_raddr), .rdata(fs_rdata));
    logic [A-1:0] ispad [0:IS-1];
    logic [IS-1:0] zbuf;
    logic signed [PSW-1:0] pspad [0:PS-1];

    // ---- control state --------------------------------------------------------------------------
    typedef enum logic [1:0] { IDLE, LOADW, RUN, DRAIN } state_t;
    state_t state;
    logic [FAW-1:0] ld_cnt;             // weights written
    logic [1:0] ld_lane;                // lane of the filter beat being written
    logic [FAW-1:0] nweights;           // p*q*S
    logic [10:0] row_w;                 // ifmap row length (F-1)*U + S
    // ifmap loader: global column counter over the rows of the pass
    logic [15:0] ld_col;                // columns fully loaded
    logic [2:0] ld_ch;                  // channel within the column being loaded
    logic [3:0] ld_slot;                // ld_col mod S
    // MAC / input-psum issue
    logic [7:0] row, x;                 // current row and output
    logic [15:0] row_base;              // global column of the current row's first column
    logic [3:0] bslot;                  // (row_base + x*U) mod S
    logic macph;                        // 1: MAC phase of output x, 0: input-psum phase
    logic mac_done;                     // every MAC of x issued (its window may be overwritten)
    logic [3:0] s_i, slot_i;            // kernel column and its ifmap slot
    logic [2:0] c_i;
    logic [4:0] f_i;
    logic [FAW-1:0] faddr;
    logic [A-1:0] a_hold;
    logic z_hold;
    logic [4*PSW-1:0] pin_reg;
    logic [1:0] beats_inflight;

    wire [15:0] need_cols = row_base + 16'(x) * 16'(cfg_u) + 16'(cfg_s);
    wire last_x = x == cfg_f - 1'b1;
    wire last_row = row == cfg_rows - 1'b1;
    // how far the loader may run: the window of x while its MACs are being issued; once they all
    // are, the window of the next output (or of the next row's first output)
    wire [15:0] ld_limit = !mac_done ? need_cols
                         : !last_x ? need_cols + 16'(cfg_u)
                         : !last_row ? row_base + 16'(row_w) + 16'(cfg_s)
                         : row_base + 16'(row_w);
    wire ld_active = (state == RUN) && (ld_col < ld_limit);
    assign iq_pop = ld_active && iq_valid;

    // MAC issue: window loaded, not every MAC done
    wire mac_issue = state == RUN && macph && !mac_done && ld_col >= need_cols;
    // input-psum issue: one op per filter; a new beat needs an input-psum beat (if enabled) and room
    // for its output beat
    wire beat_first = f_i[1:0] == 2'd0;
    wire room = 32'(oq_count) + 32'(beats_inflight) < OFIFO_D;
    wire acc_issue = state == RUN && !macph && (!beat_first || ((!cfg_pin || pq_valid) && room));
    assign pq_pop = acc_issue && beat_first && cfg_pin;
    wire beat_last = f_i[1:0] == 2'd3 || f_i == cfg_p - 1'b1;
    wire [4*PSW-1:0] pin_beat = beat_first ? pq_data : pin_reg;

    // S0: spad access
    wire [IAW-1:0] iaddr = IAW'(c_i) * IAW'(cfg_s) + IAW'(slot_i);
    wire [A-1:0] a0 = f_i == '0 ? ispad[iaddr] : a_hold;
    wire z0 = f_i == '0 ? zbuf[iaddr] : z_hold;
    assign fs_re = mac_issue && !z0;
    assign fs_raddr = faddr;

    // filter load: one weight per cycle from the head beat of the filter FIFO
    assign fs_we = state == LOADW && fq_valid;
    assign fs_waddr = ld_cnt;
    assign fs_wdata = fq_data[ld_lane*W +: W];
    wire ld_last = ld_cnt == nweights - 1'b1;
    assign fq_pop = fs_we && (ld_lane == 2'd3 || ld_last);

    // ---- pipeline registers -----------------------------------------------------------------------
    logic v1, mac1, z1, first1, bl1;
    logic [4:0] f1;
    logic [A-1:0] a1;
    logic signed [PSW-1:0] pin1;
    logic v2, mac2, z2, first2, bl2;
    logic [4:0] f2;
    logic signed [W+A:0] prod2;
    logic signed [PSW-1:0] pin2;
    logic signed [PSW-1:0] ostage [0:3];
    wire pipe_empty = !v1 && !v2;

    // S2 result of an input-psum op
    wire signed [PSW-1:0] acc_val = pspad[f2] + pin2;
    always_comb begin
        oq_data = '0;
        for (int k = 0; k < 4; k++)
            oq_data[k*PSW +: PSW] = (k == int'(f2[1:0])) ? acc_val : (k < int'(f2[1:0]) ? ostage[k] : '0);
    end
    assign oq_push = v2 && !mac2 && bl2;
    assign busy = state != IDLE;

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            ld_cnt <= '0; ld_lane <= '0; nweights <= '0; row_w <= '0;
            ld_col <= '0; ld_ch <= '0; ld_slot <= '0;
            row <= '0; x <= '0; row_base <= '0; bslot <= '0; macph <= 1'b1; mac_done <= 1'b0;
            s_i <= '0; slot_i <= '0; c_i <= '0; f_i <= '0; faddr <= '0;
            a_hold <= '0; z_hold <= 1'b0; pin_reg <= '0; beats_inflight <= '0;
            v1 <= 1'b0; mac1 <= 1'b0; z1 <= 1'b0; first1 <= 1'b0; bl1 <= 1'b0; f1 <= '0; a1 <= '0; pin1 <= '0;
            v2 <= 1'b0; mac2 <= 1'b0; z2 <= 1'b0; first2 <= 1'b0; bl2 <= 1'b0; f2 <= '0; prod2 <= '0; pin2 <= '0;
            zbuf <= '0;
            for (int k = 0; k < PS; k++) pspad[k] <= '0;
            for (int k = 0; k < IS; k++) ispad[k] <= '0;
            for (int k = 0; k < 4; k++) ostage[k] <= '0;
        end else begin
            // ---- filter load ----
            if (state == IDLE && start) begin
                state <= LOADW;
                ld_cnt <= '0; ld_lane <= '0;
                nweights <= FAW'(cfg_p) * FAW'(cfg_q) * FAW'(cfg_s);
                row_w <= (11'(cfg_f) - 11'd1) * 11'(cfg_u) + 11'(cfg_s);
            end
            if (fs_we) begin
                ld_cnt <= ld_cnt + 1'b1;
                ld_lane <= fq_pop ? 2'd0 : ld_lane + 1'b1;
                if (ld_last) begin
                    state <= RUN;
                    ld_col <= '0; ld_ch <= '0; ld_slot <= '0;
                    row <= '0; x <= '0; row_base <= '0; bslot <= '0; macph <= 1'b1; mac_done <= 1'b0;
                    s_i <= '0; slot_i <= '0; c_i <= '0; f_i <= '0; faddr <= '0;
                end
            end
            // ---- ifmap loader: column by column, channel inside column ----
            if (iq_pop) begin
                ispad[IAW'(ld_ch) * IAW'(cfg_s) + IAW'(ld_slot)] <= iq_data;
                zbuf[IAW'(ld_ch) * IAW'(cfg_s) + IAW'(ld_slot)] <= iq_data == '0;
                if (ld_ch == cfg_q - 1'b1) begin
                    ld_ch <= '0;
                    ld_col <= ld_col + 1'b1;
                    ld_slot <= ld_slot == cfg_s - 1'b1 ? '0 : ld_slot + 1'b1;
                end else ld_ch <= ld_ch + 1'b1;
            end
            // ---- MAC issue: for s, for c, for f ----
            if (mac_issue) begin
                if (f_i == '0) begin a_hold <= a0; z_hold <= z0; end
                faddr <= faddr + 1'b1;
                if (f_i == cfg_p - 1'b1) begin
                    f_i <= '0;
                    if (c_i == cfg_q - 1'b1) begin
                        c_i <= '0;
                        if (s_i == cfg_s - 1'b1) begin
                            s_i <= '0; slot_i <= bslot; mac_done <= 1'b1; macph <= 1'b0; faddr <= '0;
                        end else begin
                            s_i <= s_i + 1'b1;
                            slot_i <= slot_i == cfg_s - 1'b1 ? '0 : slot_i + 1'b1;
                        end
                    end else c_i <= c_i + 1'b1;
                end else f_i <= f_i + 1'b1;
            end
            // ---- input-psum issue: one op per filter, then the next output ----
            if (acc_issue) begin
                if (beat_first && cfg_pin) pin_reg <= pq_data;
                if (f_i == cfg_p - 1'b1) begin
                    f_i <= '0; macph <= 1'b1; mac_done <= 1'b0;
                    if (!last_x) begin
                        x <= x + 1'b1;
                        bslot <= 32'(bslot) + 32'(cfg_u) >= 32'(cfg_s) ? bslot + cfg_u - cfg_s : bslot + cfg_u;
                        slot_i <= 32'(bslot) + 32'(cfg_u) >= 32'(cfg_s) ? bslot + cfg_u - cfg_s : bslot + cfg_u;
                    end else begin
                        x <= '0;
                        row_base <= row_base + 16'(row_w);
                        slot_i <= bslot;          // (row_base + row_w) mod S == bslot of the last output
                        if (last_row) state <= DRAIN;
                        else row <= row + 1'b1;
                    end
                end else f_i <= f_i + 1'b1;
            end
            if (state == DRAIN && pipe_empty && beats_inflight == '0) state <= IDLE;
            beats_inflight <= beats_inflight + 2'(acc_issue && beat_first) - 2'(oq_push);
            // ---- S0 -> S1 ----
            v1 <= mac_issue || acc_issue;
            mac1 <= mac_issue;
            if (mac_issue || acc_issue) begin
                f1 <= f_i;
                first1 <= s_i == '0 && c_i == '0;
                z1 <= mac_issue && z0;
                bl1 <= beat_last;
                if (mac_issue && !z0) a1 <= a0;               // gated: operand registers hold
                if (acc_issue) pin1 <= cfg_pin ? $signed(pin_beat[f_i[1:0]*PSW +: PSW]) : '0;
            end
            // ---- S1: multiply ----
            v2 <= v1;
            mac2 <= mac1;
            if (v1) begin
                f2 <= f1; first2 <= first1; z2 <= z1; bl2 <= bl1; pin2 <= pin1;
                if (mac1 && !z1) prod2 <= $signed(fs_rdata) * $signed({1'b0, a1});
            end
            // ---- S2: add ----
            if (v2 && mac2) begin
                if (z2) begin
                    if (first2) pspad[f2] <= '0;
                end else
                    pspad[f2] <= (first2 ? '0 : pspad[f2]) + PSW'(prod2);
            end
            if (v2 && !mac2) begin
                ostage[f2[1:0]] <= acc_val;
            end
        end
    end

    // synthesis translate_off
    always_ff @(posedge clk) if (!reset) begin
        assert (!(oq_push && !oq_ready)) else $fatal(1, "output psum FIFO overflow");
        assert (!(start && state != IDLE)) else $fatal(1, "start while busy");
        assert (!(mac_issue && acc_issue)) else $fatal(1, "two issues in one cycle");
    end
    // synthesis translate_on
endmodule

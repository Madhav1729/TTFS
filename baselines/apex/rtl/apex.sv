// APEX (LoAS TPPE array + PASC-IF) at T = 15 with 16 INT4 TPPEs on one B bus.
// Data-A and compressor words are packed back to back; PASC stages are PC-cycle paths.

// ---- data-B buffer: the chunk's non-zero weights (BLOCK x WB bits), 64-bit write, WB-bit read ----
module apex_dbufb #(parameter int WB = 4, BLOCK = 128) (
    input  logic                      clk,
    input  logic                      we,
    input  logic [$clog2(BLOCK*WB/64)-1:0] waddr,
    input  logic [63:0]               wdata,
    input  logic [$clog2(BLOCK)-1:0]  raddr,
    output logic signed [WB-1:0]      rdata
);
    localparam int BEATS = BLOCK * WB / 64;
    logic [63:0] mem [BEATS];
    always_ff @(posedge clk) if (we) mem[waddr] <= wdata;
    localparam int PER = 64 / WB;
    logic [63:0] beat;
    assign beat = mem[32'(raddr) / PER];
    assign rdata = beat[(32'(raddr) % PER) * WB +: WB];
endmodule

// ---- data-A buffer: two contexts x BLOCK words of T bits, 64-bit write, T-bit read ---------------
// Words are packed back to back, so a read spans two adjacent beats (the second clamped at the end).
module apex_dbufa #(parameter int T = 8, BLOCK = 128) (
    input  logic                      clk,
    input  logic                      we,
    input  logic                      wctx,
    input  logic [$clog2((BLOCK*T+63)/64)-1:0] waddr,
    input  logic [63:0]               wdata,
    input  logic                      rctx,
    input  logic [$clog2(BLOCK)-1:0]  raddr,
    output logic [T-1:0]              rdata
);
    localparam int BEATS = (BLOCK * T + 63) / 64;
    localparam int BW = $clog2(BEATS);
    localparam int SB = $clog2(BLOCK * T);       // stream bit position
    logic [63:0] mem [2][BEATS];
    always_ff @(posedge clk) if (we) mem[wctx][waddr] <= wdata;
    logic [SB-1:0] bp;
    logic [BW-1:0] b, bh;
    logic [5:0]    o;
    logic [127:0]  win;
    assign bp = SB'(raddr) * SB'(T);
    assign b  = BW'(bp >> 6);
    assign o  = bp[5:0];
    assign bh = (32'(b) + 1 < BEATS) ? b + 1'b1 : b;
    assign win = {mem[rctx][bh], mem[rctx][b]};
    assign rdata = win[7'(o) +: T];
endmodule

// ---- PASC-IF: three unrolled stages, each a PC-cycle multicycle path (synth/apex_mcp.tcl) ----
module apex_pasc #(parameter int T = 8, OW = 15, MW = 19, FB = 4, PC = 4) (
    input  logic                  clk,
    input  logic                  rst,
    input  logic                  in_valid,
    input  logic signed [OW-1:0]  o [T],        // post-synaptic input per timestep (held by the TPPE)
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [MW-1:0]         theta,        // theta* (fixed point, FB fractional bits; < 2^(MW-1))
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [MW-1:0]         mem0,         // initial membrane: theta*/2 + folded bias (two's complement)
    input  logic                  flip,         // channel with a negative BN/step affine
    output logic                  busy1,        // stage 1 still reads o
    output logic                  out_valid,
    output logic [T-1:0]          spikes        // output spike train (bit t = spike at t)
);
    localparam int SW = $clog2(2 * T) + 1;      // spike count, signed: -(T-1) .. 2T-1
    localparam int CB = $clog2(PC);
    logic signed [MW-1:0] p_th, p_m0;
    logic                 p_flip;
    // stage 1: excitatory accumulation over T timesteps, soft reset
    logic signed [MW-1:0] x [T];
    logic signed [MW-1:0] m1;
    logic signed [SW-1:0] s1;
    always_comb begin
        for (int t = 0; t < T; t++)
            x[t] = (p_flip ? -MW'(o[t]) : MW'(o[t])) <<< FB;
        m1 = p_m0; s1 = '0;
        for (int t = 0; t < T; t++) begin
            m1 = m1 + x[t];
            if (m1 >= p_th) begin s1 = s1 + 1'b1; m1 = m1 - p_th; end
        end
    end
    logic signed [MW-1:0] r_m1, r_th;
    logic signed [SW-1:0] r_s1, r_s2;
    // stage 2: inhibitory correction over max(L_{n-1}, L_n) - 1 timesteps
    logic signed [MW-1:0] m2;
    logic signed [SW-1:0] s2;
    always_comb begin
        m2 = r_m1; s2 = r_s1;
        for (int t = 0; t < T - 1; t++) begin
            if (m2 >= r_th) begin s2 = s2 + 1'b1; m2 = m2 - r_th; end
            else if (m2 < 0) begin s2 = s2 - 1'b1; m2 = m2 + r_th; end
        end
    end
    // stage 3: output spike generation over L_n timesteps (membrane reset to the spike count)
    logic signed [SW-1:0] m3;
    logic [T-1:0] sp3;
    always_comb begin
        m3 = r_s2; sp3 = '0;
        for (int t = 0; t < T; t++)
            if (m3 >= 1) begin sp3[t] = 1'b1; m3 = m3 - 1'b1; end
    end
    logic b1, b2, b3;
    logic [CB-1:0] c1, c2, c3;
    assign busy1 = b1;
    always_ff @(posedge clk) begin
        if (rst) begin b1 <= 1'b0; b2 <= 1'b0; b3 <= 1'b0; out_valid <= 1'b0; end
        else begin
            out_valid <= 1'b0;
            if (in_valid) begin
                b1 <= 1'b1; c1 <= '0;
                p_th <= $signed({1'b0, theta[MW-2:0]}); p_m0 <= $signed(mem0); p_flip <= flip;
            end else if (b1) begin
                c1 <= c1 + 1'b1;
                if (c1 == CB'(PC - 1)) begin b1 <= 1'b0; r_m1 <= m1; r_s1 <= s1; r_th <= p_th; b2 <= 1'b1; c2 <= '0; end
            end
            if (b2) begin
                c2 <= c2 + 1'b1;
                if (c2 == CB'(PC - 1)) begin r_s2 <= s2; b2 <= 1'b0; b3 <= 1'b1; c3 <= '0; end
            end
            if (b3) begin
                c3 <= c3 + 1'b1;
                if (c3 == CB'(PC - 1)) begin spikes <= sp3; out_valid <= 1'b1; b3 <= 1'b0; end
            end
        end
    end
    // outputs enter >= PC + 2 cycles apart (the TPPE waits for busy1), so a stage is always free
    // when the previous one hands over
    always_ff @(posedge clk) if (!rst) begin
        if (in_valid) assert (!b1);
        if (b1 && c1 == CB'(PC - 1)) assert (!b2);
        if (b2 && c2 == CB'(PC - 1)) assert (!b3);
    end
endmodule

// ---- one TPPE -----------------------------------------------------------------------------------
module apex_tppe #(parameter int WB = 4, T = 8, BLOCK = 128, FD = 8, PW = 14, CW = 14, MW = 19, FB = 4, PC = 4) (
    input  logic        clk,
    input  logic        rst,
    // B side (broadcast): addr 0/1 = bm-B halves, 2.. = data-B beats
    input  logic        b_we,
    input  logic [4:0]  b_addr,
    input  logic [63:0] b_data,
    // A side (own crossbar port): addr 0/1 = bm-A halves, 2.. = data-A beats of context a_ctx
    // (apex15: 6-bit address, 2 + 30 data-A beats at T = 15 and room for T = 16)
    input  logic        a_we,
    input  logic [5:0]  a_addr,
    input  logic        a_ctx,
    input  logic [63:0] a_data,
    input  logic        a_done,             // data-A of context a_ctx complete
    // PASC parameters of the next output channel
    input  logic        p_we,
    input  logic [MW-1:0] p_theta,
    input  logic [MW-1:0] p_mem0,
    input  logic        p_flip,
    // chunk start (bm-A, bm-B, data-B loaded)
    input  logic        go,
    input  logic        go_first,
    input  logic        go_last,
    output logic        ready,              // go would be accepted now
    output logic        b_free,             // fast path idle: bm-B / data-B may be rewritten
    output logic        next_ctx,           // context the next chunk with matches will use
    output logic        has_match,          // (bm-A & bm-B) != 0 for the loaded masks
    output logic        out_valid,
    output logic [T-1:0] out_spikes,
    // activity
    output logic        st_match,
    output logic        st_corr
);
    localparam int IW = $clog2(BLOCK);
    localparam int OW = (PW > CW ? PW : CW) + 1;
    localparam int G = BLOCK / 16;

    logic [BLOCK-1:0] bm_a, bm_b, andr;
    always_ff @(posedge clk) begin
        if (a_we && a_addr == 6'd0) bm_a[63:0] <= a_data;
        if (a_we && a_addr == 6'd1) bm_a[BLOCK-1:64] <= a_data;
        if (b_we && b_addr == 5'd0) bm_b[63:0] <= b_data;
        if (b_we && b_addr == 5'd1) bm_b[BLOCK-1:64] <= b_data;
    end
    assign has_match = |(bm_a & bm_b);

    // data buffers (SRAM-like: subtracted from the logic power, costed with CACTI)
    logic [IW-1:0] woff;
    logic signed [WB-1:0] wval;
    apex_dbufb #(.WB(WB), .BLOCK(BLOCK)) dbufb (
        .clk, .we(b_we && b_addr >= 5'd2), .waddr(($clog2(BLOCK*WB/64))'(b_addr - 5'd2)), .wdata(b_data),
        .raddr(woff), .rdata(wval));
    logic c_ctx;
    logic [IW-1:0] aoff;
    logic [T-1:0] aword;
    apex_dbufa #(.T(T), .BLOCK(BLOCK)) dbufa (
        .clk, .we(a_we && a_addr >= 6'd2), .wctx(a_ctx), .waddr(($clog2((BLOCK*T+63)/64))'(a_addr - 6'd2)), .wdata(a_data),
        .rctx(c_ctx), .raddr(aoff), .rdata(aword));

    // ---- contexts: laggy prefix state + data-A, two deep --------------------------------------
    logic [BLOCK-1:0] lag_bm [2];
    logic [7:0]       base [2][G];
    logic [1:0]       busy, lready, dvalid;
    logic [7:0]       outstanding [2];
    logic             alloc;                 // next context to allocate
    assign next_ctx = alloc;

    // ---- PASC parameters ----------------------------------------------------------------------
    logic [MW-1:0] st_theta, st_mem0, ac_theta, ac_mem0;
    logic          st_flip, ac_flip;
    always_ff @(posedge clk) if (p_we) begin st_theta <= p_theta; st_mem0 <= p_mem0; st_flip <= p_flip; end

    // ---- fast path: AND -> priority encoder -> fast prefix sum -> data-B -> pseudo-accumulator --
    logic f1_v;
    logic [IW-1:0] f1_pos, f1_woff;
    logic f1_ctx, cur_ctx;
    logic [$clog2(FD+1)-1:0] fcount;         // FIFO occupancy
    logic last_pending, firing;
    logic fast_busy;
    assign fast_busy = (|andr) || f1_v;
    assign b_free = !fast_busy;
    logic [IW-1:0] pe_pos;
    always_comb begin
        pe_pos = '0;
        for (int i = BLOCK - 1; i >= 0; i--) if (andr[i]) pe_pos = IW'(i);
    end
    function automatic logic [IW:0] popcount(input logic [BLOCK-1:0] v);
        popcount = '0;
        for (int i = 0; i < BLOCK; i++) popcount = popcount + (IW+1)'(v[i]);
    endfunction
    function automatic logic [4:0] pop16(input logic [15:0] v);
        pop16 = '0;
        for (int i = 0; i < 16; i++) pop16 = pop16 + 5'(v[i]);
    endfunction
    logic fifo_space;
    assign fifo_space = (32'(fcount) + 32'(f1_v)) < FD;
    logic take;
    assign take = (|andr) && fifo_space;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [IW:0] fast_off;                   // 0..127 here (the matched bit itself is excluded)
    /* verilator lint_on UNUSEDSIGNAL */
    assign woff = f1_woff;
    assign fast_off = popcount(bm_b & ((BLOCK'(1) << pe_pos) - BLOCK'(1)));

    logic acc_free;
    assign acc_free = !last_pending && !firing;
    logic go_ok;
    assign go_ok = !fast_busy && (!go_first || acc_free) && (!has_match || !busy[alloc]) && !last_pending;
    assign ready = go_ok;

    // ---- FIFO-mp {ctx, pos} and FIFO-B {w} ----------------------------------------------------
    logic [IW:0]      fifo_mp [FD];
    logic signed [WB-1:0] fifo_b [FD];
    logic [$clog2(FD)-1:0] fhead, ftail;
    logic push, pop;

    // ---- laggy prefix sum: one 16-bit group per cycle (16 adders), 8 cycles per chunk ----------
    logic        lag_run, lag_ctx, lag_next;   // contexts are processed in allocation order
    logic [2:0]  lag_g;
    logic [7:0]  lag_acc;

    // ---- corrections ---------------------------------------------------------------------------
    logic [IW-1:0] h_pos;
    logic          h_ctx;
    assign h_ctx = fifo_mp[fhead][IW];
    assign h_pos = fifo_mp[fhead][IW-1:0];
    assign c_ctx = h_ctx;
    assign aoff = IW'(base[h_ctx][h_pos / 16] + 8'(pop16(lag_bm[h_ctx][(h_pos / 16) * 16 +: 16] & ((16'(1) << (h_pos % 16)) - 16'(1)))));
    assign pop = (fcount != 0) && lready[h_ctx] && dvalid[h_ctx];
    logic c1_v;
    logic [T-1:0] c1_word;
    logic signed [WB-1:0] c1_w;

    logic signed [PW-1:0] pseudo;
    logic signed [CW-1:0] corr [T];
    logic signed [OW-1:0] o_cap [T];
    logic cap_v, pasc_busy1;
    logic drained;
    assign drained = !fast_busy && fcount == 0 && !c1_v;
    assign firing = cap_v;

    assign push = f1_v;
    assign st_match = f1_v;
    assign st_corr = pop;

    always_ff @(posedge clk) begin
        if (rst) begin
            andr <= '0; f1_v <= 1'b0; fcount <= '0; fhead <= '0; ftail <= '0;
            busy <= '0; lready <= '0; dvalid <= '0; alloc <= 1'b0;
            lag_next <= 1'b0; lag_run <= 1'b0; c1_v <= 1'b0; cap_v <= 1'b0; last_pending <= 1'b0;
            pseudo <= '0;
            for (int t = 0; t < T; t++) corr[t] <= '0;
        end else begin
            // chunk start
            if (go) begin
                andr <= bm_a & bm_b;
                cur_ctx <= alloc;
                if (go_first) begin ac_theta <= st_theta; ac_mem0 <= st_mem0; ac_flip <= st_flip; end
                if (go_last) last_pending <= 1'b1;
                if (has_match) begin
                    busy[alloc] <= 1'b1; lready[alloc] <= 1'b0;
                    lag_bm[alloc] <= bm_a;
                    outstanding[alloc] <= 8'(popcount(bm_a & bm_b));
                    alloc <= ~alloc;
                end
            end
            // fast path stage 1: priority encoder + fast prefix sum
            f1_v <= take;
            if (take) begin
                f1_pos <= pe_pos; f1_woff <= fast_off[IW-1:0]; f1_ctx <= cur_ctx;
                andr[pe_pos] <= 1'b0;
            end
            // fast path stage 2: data-B read, pseudo-accumulate, push FIFOs
            if (f1_v) begin
                pseudo <= pseudo + PW'(wval);
                fifo_mp[ftail] <= {f1_ctx, f1_pos};
                fifo_b[ftail] <= wval;
                ftail <= ftail + 1'b1;
            end
            // laggy prefix sum
            if (!lag_run && busy[lag_next] && !lready[lag_next]) begin
                lag_run <= 1'b1; lag_ctx <= lag_next; lag_g <= '0; lag_acc <= '0;
                lag_next <= ~lag_next;
            end else if (lag_run) begin
                base[lag_ctx][lag_g] <= lag_acc;
                lag_acc <= lag_acc + 8'(pop16(lag_bm[lag_ctx][lag_g * 16 +: 16]));
                lag_g <= lag_g + 1'b1;
                if (lag_g == 3'(G - 1)) begin lag_run <= 1'b0; lready[lag_ctx] <= 1'b1; end
            end
            // data-A arrival
            if (a_done) dvalid[a_ctx] <= 1'b1;
            // correction stage 1: laggy offset -> data-A read, pop FIFOs
            c1_v <= pop;
            if (pop) begin
                c1_word <= aword; c1_w <= fifo_b[fhead];
                fhead <= fhead + 1'b1;
                outstanding[h_ctx] <= outstanding[h_ctx] - 1'b1;
                if (outstanding[h_ctx] == 8'd1) begin busy[h_ctx] <= 1'b0; lready[h_ctx] <= 1'b0; dvalid[h_ctx] <= 1'b0; end
            end
            // correction stage 2: correction accumulator of every timestep the input did not fire
            if (c1_v)
                for (int t = 0; t < T; t++) if (!c1_word[t]) corr[t] <= corr[t] + CW'(c1_w);
            fcount <= fcount + $bits(fcount)'(push) - $bits(fcount)'(pop);
            // output neuron complete: O[t] = pseudo - corr[t] to the PASC-IF unit
            cap_v <= 1'b0;
            if (last_pending && drained && !go && !pasc_busy1 && !cap_v) begin
                for (int t = 0; t < T; t++) o_cap[t] <= OW'(pseudo) - OW'(corr[t]);
                cap_v <= 1'b1;
                last_pending <= 1'b0;
                pseudo <= '0;
                for (int t = 0; t < T; t++) corr[t] <= '0;
            end
        end
    end

    apex_pasc #(.T(T), .OW(OW), .MW(MW), .FB(FB), .PC(PC)) pasc (
        .clk, .rst, .in_valid(cap_v), .o(o_cap), .theta(ac_theta), .mem0(ac_mem0), .flip(ac_flip),
        .busy1(pasc_busy1), .out_valid, .spikes(out_spikes));
endmodule

// ---- periphery: two 16x16 swizzle-switch crossbars + compressor ---------------------------------
module apex_periphery #(parameter int NT = 16, T = 8, BLOCK = 128) (
    input  logic        clk,
    input  logic        rst,
    // global-cache banks -> TPPE A ports (registered crossbar)
    input  logic [63:0] bank_rdata [NT],
    input  logic [$clog2(NT)-1:0] a_sel [NT],
    input  logic        a_we_in [NT],
    input  logic [5:0]  a_addr_in [NT],
    input  logic        a_ctx_in [NT],
    input  logic        a_done_in [NT],
    output logic [63:0] a_data [NT],
    output logic        a_we [NT],
    output logic [5:0]  a_addr [NT],
    output logic        a_ctx [NT],
    output logic        a_done [NT],
    // TPPE outputs -> compressor (one row unit per TPPE)
    input  logic        o_valid [NT],
    input  logic [T-1:0] o_spikes [NT],
    input  logic [9:0]  m_cfg,              // output channels of the layer
    input  logic        row_reset,
    // compressor -> banks (registered crossbar)
    input  logic [$clog2(NT)-1:0] w_src [NT],
    output logic        bank_we [NT],
    output logic [63:0] bank_wdata [NT]
);
    // crossbar 1 (data / address registers load only with a transfer, so they are clock-gated)
    always_ff @(posedge clk) begin
        for (int p = 0; p < NT; p++) begin
            if (a_we_in[p]) begin a_data[p] <= bank_rdata[a_sel[p]]; a_addr[p] <= a_addr_in[p]; end
            if (a_we_in[p] || a_done_in[p]) a_ctx[p] <= a_ctx_in[p];
            a_we[p] <= rst ? 1'b0 : a_we_in[p];
            a_done[p] <= rst ? 1'b0 : a_done_in[p];
        end
    end
    // compressor rows: pack the non-silent T-bit words into 64-bit beats, then bitmask and pointer
    // per 128-channel fiber; one output can push up to 5 beats, so the row FIFO is 8 deep.
    localparam int QD = 8;
    logic        q_v [NT];
    logic [63:0] q_beat [NT];
    for (genvar r = 0; r < NT; r++) begin : rows
        logic [BLOCK-1:0] mask, fm;
        logic [6:0]       col;
        logic [9:0]       total;
        logic [6:0]       fill;                 // bits used in pack, 0..63
        logic [63:0]      pack, rem;
        logic [64+T-1:0]  comb;                 // pack with the new word appended
        logic [6:0]       rem_bits;
        logic [31:0]      ptr, fstart;          // write address; start of the current fiber's data
        logic [63:0]      fifo [QD];
        logic [$clog2(QD)-1:0] h, t;
        logic [$clog2(QD+1)-1:0] n;
        logic [2:0]       pushes;
        logic [63:0]      pb [5];
        logic             last_col, nz, full;
        always_comb begin
            last_col = (col == 7'(BLOCK - 1)) || (total + 1'b1 == m_cfg);
            nz = |o_spikes[r];
            comb = (64+T)'(pack) | ((64+T)'(o_spikes[r]) << fill);
            full = nz && (32'(fill) + T >= 64);
            // what stays in the packer after this output
            if (!nz)       begin rem = pack;               rem_bits = fill; end
            else if (full) begin rem = 64'(comb[64 +: T]); rem_bits = 7'(32'(fill) + T - 64); end
            else           begin rem = comb[63:0];         rem_bits = 7'(32'(fill) + T); end
            fm = mask; fm[col] = nz;
            pushes = '0;
            for (int k = 0; k < 5; k++) pb[k] = '0;
            if (o_valid[r]) begin
                if (full) begin pb[pushes] = comb[63:0]; pushes = pushes + 1'b1; end
                if (last_col) begin
                    if (rem_bits != 0) begin pb[pushes] = rem; pushes = pushes + 1'b1; end
                    pb[pushes] = fm[63:0]; pushes = pushes + 1'b1;
                    pb[pushes] = fm[BLOCK-1:64]; pushes = pushes + 1'b1;
                    pb[pushes] = {32'd0, fstart}; pushes = pushes + 1'b1;
                end
            end
        end
        always_ff @(posedge clk) begin
            if (rst || row_reset) begin
                mask <= '0; col <= '0; total <= '0; fill <= '0; pack <= '0; ptr <= '0; fstart <= '0;
                h <= '0; t <= '0; n <= '0;
            end else begin
                if (o_valid[r]) begin
                    total <= total + 1'b1;
                    ptr <= ptr + 32'(pushes);
                    if (last_col) begin
                        col <= '0; mask <= '0; fill <= '0; pack <= '0; fstart <= ptr + 32'(pushes);
                    end else begin
                        col <= col + 1'b1; mask <= fm; fill <= rem_bits; pack <= rem;
                    end
                end
                for (int k = 0; k < 5; k++)
                    if (3'(k) < pushes) fifo[t + $clog2(QD)'(k)] <= pb[k];
                t <= t + $clog2(QD)'(pushes);
                n <= n + $clog2(QD+1)'(pushes) - $clog2(QD+1)'(n != 0);
                if (n != 0) h <= h + 1'b1;
            end
        end
        always_ff @(posedge clk) if (!rst && !row_reset) assert (32'(n) - 32'(n != 0) + 32'(pushes) <= QD);
        assign q_v[r] = n != 0;
        assign q_beat[r] = fifo[h];
    end
    // crossbar 2
    always_ff @(posedge clk) begin
        for (int b = 0; b < NT; b++) begin
            bank_we[b] <= rst ? 1'b0 : q_v[w_src[b]];
            if (q_v[w_src[b]]) bank_wdata[b] <= q_beat[w_src[b]];
        end
    end
endmodule

// ---- top: periphery + NT INT4 TPPEs on one B broadcast bus ---------------------------------------
module apex_top #(parameter int NT = 16, T = 15, BLOCK = 128, PW = 12, CW = 12, MW = 18, FB = 6, PC = 8) (
    input  logic        clk,
    input  logic        rst,
    // B broadcast
    input  logic        b_we,
    input  logic [4:0]  b_addr,
    input  logic [63:0] b_data,
    input  logic        p_we,
    input  logic [MW-1:0] p_theta,
    input  logic [MW-1:0] p_mem0,
    input  logic        p_flip,
    input  logic        go,
    input  logic        go_first,
    input  logic        go_last,
    // global-cache side
    input  logic [63:0] bank_rdata [NT],
    input  logic [$clog2(NT)-1:0] a_sel [NT],
    input  logic        a_we_in [NT],
    input  logic [5:0]  a_addr_in [NT],
    input  logic        a_ctx_in [NT],
    input  logic        a_done_in [NT],
    input  logic [9:0]  m_cfg,
    input  logic        row_reset,
    input  logic [$clog2(NT)-1:0] w_src [NT],
    output logic        bank_we [NT],
    output logic [63:0] bank_wdata [NT],
    // status
    output logic [NT-1:0] ready,
    output logic [NT-1:0] b_free,
    output logic [NT-1:0] next_ctx,
    output logic [NT-1:0] has_match,
    output logic [NT-1:0] out_valid,
    output logic [T-1:0]  out_spikes [NT],
    output logic [NT-1:0] st_match,
    output logic [NT-1:0] st_corr
);
    logic [63:0] a_data [NT];
    logic        a_we [NT], a_ctx [NT], a_done [NT];
    logic [5:0]  a_addr [NT];
    logic        ov [NT];
    logic [T-1:0] os [NT];
    apex_periphery #(.NT(NT), .T(T), .BLOCK(BLOCK)) peri (
        .clk, .rst, .bank_rdata, .a_sel, .a_we_in, .a_addr_in, .a_ctx_in, .a_done_in,
        .a_data, .a_we, .a_addr, .a_ctx, .a_done, .o_valid(ov), .o_spikes(os), .m_cfg, .row_reset,
        .w_src, .bank_we, .bank_wdata);
    for (genvar i = 0; i < NT; i++) begin : t4
        apex_tppe #(.WB(4), .T(T), .BLOCK(BLOCK), .PW(PW), .CW(CW), .MW(MW), .FB(FB), .PC(PC)) pe (
            .clk, .rst, .b_we, .b_addr, .b_data,
            .a_we(a_we[i]), .a_addr(a_addr[i]), .a_ctx(a_ctx[i]), .a_data(a_data[i]), .a_done(a_done[i]),
            .p_we, .p_theta, .p_mem0, .p_flip, .go, .go_first, .go_last,
            .ready(ready[i]), .b_free(b_free[i]), .next_ctx(next_ctx[i]), .has_match(has_match[i]),
            .out_valid(out_valid[i]), .out_spikes(out_spikes[i]), .st_match(st_match[i]), .st_corr(st_corr[i]));
    end
    always_comb for (int i = 0; i < NT; i++) begin ov[i] = out_valid[i]; os[i] = out_spikes[i]; end
endmodule

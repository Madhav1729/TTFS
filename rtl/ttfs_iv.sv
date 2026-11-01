// TTFS-64 tile with current/membrane (S/U) lanes, lane-skip issue and a shared threshold table.

// One output channel: S += w per spike, U += S per closed tick, threshold compare.
module ttfs_iv_lane #(parameter integer W = 4, B = 13, ACC = 32, T = 8, WS = 16, TW = 16, LS = 1, GATE = 1,
                      TIME = $clog2(T), TC = $clog2(T+1), WP = $clog2(WS), UB = B + TC) (
    input  logic clk, reset, active,
    input  logic [WS-1:0] land,                // slot j's weight column arrives this cycle ...
    input  logic [WS-1:0] land_nz,             // ... with a non-zero weight for this lane
    input  logic [WP-1:0] head,                // oldest slot of the window
    input  logic [WS*W-1:0] sel_w,             // this lane's weight in every slot
    input  logic [WS*TIME-1:0] sel_t,          // every slot's tick
    input  logic [TC-1:0] t_safe,              // no spike of a smaller tick can still land
    output logic [WS-1:0] pend,
    output logic picked,
    output logic done,                         // all ticks closed, nothing pending
    input  logic pass_start, pass_end, ramp_first,
    input  logic [ACC-1:0] old_v,
    output logic [ACC-1:0] new_v,
    input  logic cmp_load, compare,
    input  logic [TIME-1:0] cmp_t,
    input  logic [ACC-1:0] cmp_v,
    input  logic [TW-1:0] th_cur,               // this lane's threshold for cmp_t, from the shared table
    input  logic down,                         // this lane's crossing direction
    output logic [ACC-1:0] result,
    output logic spike,
    output logic do_add
);
    // oldest pending slot (rotating priority from the window head), its weight and tick
    logic [WS-1:0] cand;
    logic [W-1:0] ew;
    logic [TC-1:0] et;
    always_comb begin
        cand = '0; ew = '0; et = '0;
        for (int k = WS - 1; k >= 0; k--)
            if (pend[WP'(head + WP'(k))]) begin cand = '0; cand[WP'(head + WP'(k))] = 1'b1; end
        for (int j = 0; j < WS; j++)
            if (cand[j]) begin ew = sel_w[j*W +: W]; et = TC'(sel_t[j*TIME +: TIME]); end
    end
    logic signed [B-1:0] s_cur;
    logic signed [UB-1:0] u_cur, uo;
    logic [TC-1:0] tc;
    wire has = |cand;
    wire can_close = tc < TC'(T) && tc < t_safe && (!has || et > tc);
    wire jump = can_close && s_cur == '0;                  // U += 0 over any number of ticks
    logic [TC-1:0] jump_to;
    always_comb begin
        jump_to = t_safe;
        if (has && et < jump_to) jump_to = et;
        if (TC'(T) < jump_to) jump_to = TC'(T);
    end
    wire take_now = has && et == tc;
    wire take_jump = jump && has && et == jump_to;         // land on the spike's tick and add it
    wire take_next = !jump && has && can_close && et == tc + 1'b1;
    // LS = 0 (in order): the oldest pending event may be taken only once it is the window's oldest event;
    // ticks still close against it, so no tick is closed past an event that has already landed
    wire in_turn = LS != 0 || cand[head];
    wire take = in_turn && (take_now || take_jump || take_next);
    assign picked = take;
    assign done = tc == TC'(T) && pend == '0;
    always_comb new_v = ramp_first ? ACC'(uo) : $signed(old_v) + ACC'(uo);
    logic signed [ACC-1:0] vreg;
    logic fired, q_take;
    assign do_add = q_take;
    assign result = vreg;
    wire signed [ACC-1:0] th = ACC'($signed(th_cur));
    wire reached = down ? vreg <= th : vreg >= th;
    // GATE = 1: registers update only when the lane state can change (one clock gate per lane)
    wire lane_land = |(land & land_nz & {WS{active}});
    wire en = GATE == 0 || reset || take || lane_land || pass_start || pass_end || jump || can_close
              || q_take || spike || cmp_load || (compare && active && !fired && reached);
    always_ff @(posedge clk) if (en) begin
        if (reset) begin
            pend <= '0; s_cur <= '0; u_cur <= '0; uo <= '0; tc <= '0; q_take <= 0;
            vreg <= '0; fired <= 0; spike <= 0;
        end else begin
            q_take <= take;
            pend <= (pend & ~(take ? cand : '0)) | (land & land_nz & {WS{active}});
            if (pass_end) uo <= u_cur;
            if (pass_start) begin
                s_cur <= '0; u_cur <= '0; tc <= '0;
            end else if (take_now) begin
                if (in_turn) s_cur <= s_cur + B'($signed(ew));
            end else if (jump) begin
                tc <= jump_to;
                if (take_jump && in_turn) s_cur <= B'($signed(ew));
            end else if (can_close) begin
                u_cur <= u_cur + UB'(s_cur);
                tc <= tc + 1'b1;
                if (take_next && in_turn) s_cur <= s_cur + B'($signed(ew));
            end
            spike <= 0;
            if (cmp_load) begin
                vreg <= $signed(cmp_v);
                fired <= 0;
            end else if (compare && active && !fired && reached) begin
                spike <= 1;
                fired <= 1;
            end
        end
    end
endmodule

// Threshold table shared by the lanes (an SRAM in a real chip).
module ttfs_thr #(parameter integer LANES = 64, ACC = 32, TW = 16, T = 8, TIME = $clog2(T)) (
    input  logic clk,
    input  logic write,
    input  logic [TIME-1:0] wt,
    input  logic [LANES*ACC-1:0] wdata,
    input  logic [LANES-1:0] wdown,
    input  logic [TIME-1:0] rt,
    output logic [LANES*TW-1:0] rdata,
    output logic [LANES-1:0] down
);
    logic [LANES*TW-1:0] rows [0:T-1];
    always_ff @(posedge clk) if (write) begin
        for (int l = 0; l < LANES; l++) rows[wt][l*TW +: TW] <= TW'(wdata[l*ACC +: ACC]);
        down <= wdown;
    end
    assign rdata = rows[rt];
    // synthesis translate_off
    always_ff @(posedge clk) if (write)
        for (int l = 0; l < LANES; l++)
            assert ($signed(wdata[l*ACC +: ACC]) == ACC'($signed(TW'(wdata[l*ACC +: ACC]))))
                else $fatal(1, "threshold does not fit in TW bits");
    // synthesis translate_on
endmodule

// Tile: event window, weight banks, 64 lanes, membranes, output compaction.
module ttfs_iv #(
    parameter integer LANES = 64, W = 4, B = 13, ACC = 32, T = 8, OB = 32, CH = 256, EDEPTH = 32, TW = 16, LS = 1,
    parameter integer GATE = 1, WS = 16, S = 4, NB = 4, PUSH = 5, LANES_IMPL = 1,
    parameter integer TIME = $clog2(T), TC = $clog2(T+1), SLOT = $clog2(OB), CIDX = $clog2(CH), LR = $clog2(LANES),
    parameter integer EW = CIDX + TIME, WP = $clog2(WS), BW = $clog2(NB), RW = CIDX - BW
) (
    input  logic clk, reset,
    input  logic [LANES-1:0] active,
    input  logic wb_write, wb_buf,
    input  logic [CIDX-1:0] wb_addr,
    input  logic [LANES-1:0] wb_mask,
    input  logic [LANES*W-1:0] wb_packed,
    input  logic [$clog2(PUSH+1)-1:0] ev_push_n,
    input  logic [PUSH*EW-1:0] ev_push_data,
    output logic [$clog2(EDEPTH):0] ev_count,
    input  logic pass_start, pass_end, pass_first, pass_set, pass_wbuf, pass_bank,
    input  logic [SLOT-1:0] pass_slot,
    input  logic [CIDX:0] pass_len,
    output logic pass_idle,
    input  logic threshold_write,
    input  logic [TIME-1:0] th_t,
    input  logic [LANES*ACC-1:0] threshold_data,
    input  logic [LANES-1:0] threshold_down,
    input  logic cmp_load, compare, cmp_bank,
    input  logic [SLOT-1:0] cmp_slot,
    input  logic [TIME-1:0] cmp_t,
    output logic ramp_busy,
    output wire [LANES*ACC-1:0] results,
    output wire [LANES-1:0] spike_mask,
    output logic out_valid,
    output logic [LANES*LR-1:0] out_idx,
    output logic [LR:0] out_count,
    output wire [LANES-1:0] do_add,
    output logic [$clog2(S+1)-1:0] st_issued,          // events retired this cycle
    output logic [$clog2(NB+1)-1:0] st_reads
);
    // ---- event FIFO ---------------------------------------------------------------------------
    logic [S*EW-1:0] evh;
    logic [$clog2(S+1)-1:0] pop_n;
    ttfs_evq #(.DEPTH(EDEPTH), .WIDTH(EW), .PUSH(PUSH), .S(S)) events (
        .clk, .reset, .push_n(ev_push_n), .push_data(ev_push_data), .pop_n, .head(evh), .count(ev_count));

    // ---- pass state ---------------------------------------------------------------------------
    logic p_on, set, rbuf, first_q, bank_q;
    logic [SLOT-1:0] slot_q;
    logic [CIDX:0] p_left;

    // ---- weight banks -------------------------------------------------------------------------
    wire [LANES*W-1:0] dense;
    ttfs_unpack #(.LANES(LANES), .W(W)) unpack (.mask(wb_mask), .packed_w(wb_packed), .dense(dense));
    logic [NB-1:0] breq;
    logic [NB*RW-1:0] baddr;
    wire [NB*LANES*W-1:0] bdata;
    ttfs_wbanks #(.CH(CH), .NB(NB), .WIDTH(LANES*W)) weights (
        .clk, .write(wb_write), .wbuf(wb_buf), .waddr(wb_addr), .wdata(dense),
        .rbuf, .read(breq), .raddr(baddr), .rdata(bdata));

    // ---- event window: WS slots in arrival order, head = oldest -------------------------------
    logic [WS-1:0] sv, shd, srd;                           // occupied / column held / read in flight
    logic [CIDX-1:0] sidx [WS];
    logic [TIME-1:0] stk [WS];
    logic [LANES*W-1:0] scol [WS];
    logic [WP-1:0] head, tail;
    logic [WP:0] count;
    logic [NB-1:0] rd_v;
    logic [WP-1:0] rd_slot [NB];
    // column reads, oldest first, one per bank
    logic [WP-1:0] req_slot [NB];
    always_comb begin
        breq = '0; baddr = '0;
        for (int b = 0; b < NB; b++) req_slot[b] = '0;
        for (int k = 0; k < WS; k++) begin
            logic [WP-1:0] j;
            j = head + WP'(k);
            if (sv[j] && !shd[j] && !srd[j] && !breq[sidx[j][BW-1:0]]) begin
                breq[sidx[j][BW-1:0]] = 1'b1;
                baddr[int'(sidx[j][BW-1:0])*RW +: RW] = sidx[j][CIDX-1:BW];
                req_slot[sidx[j][BW-1:0]] = j;
            end
        end
    end
    // columns arriving this cycle, per slot
    logic [WS-1:0] land;
    logic [LANES*W-1:0] lcol [WS];
    always_comb begin
        land = '0;
        for (int j = 0; j < WS; j++) lcol[j] = '0;
        for (int b = 0; b < NB; b++)
            if (rd_v[b]) begin land[rd_slot[b]] = 1'b1; lcol[rd_slot[b]] = bdata[b*LANES*W +: LANES*W]; end
    end
    // a slot is finished once its column is held and no lane has it pending
    wire [WS-1:0] pend_l [LANES];
    logic [WS-1:0] pend_any;
    always_comb begin
        pend_any = '0;
        for (int l = 0; l < LANES; l++) pend_any = pend_any | pend_l[l];
    end
    wire [WS-1:0] fin = sv & shd & ~pend_any;
    // retire finished slots from the head, up to S
    logic [$clog2(S+1)-1:0] n_ret;
    always_comb begin
        logic stop;
        n_ret = '0; stop = 1'b0;
        for (int k = 0; k < S; k++)
            if (!stop && fin[WP'(head + WP'(k))] && k < int'(count)) n_ret = n_ret + 1'b1;
            else stop = 1'b1;
    end
    // refill at the tail from the FIFO, only this pass's events
    logic [$clog2(S+1)-1:0] n_fill;
    always_comb begin
        int f;
        f = S;
        if (f > int'(ev_count)) f = int'(ev_count);
        if (f > int'(p_left)) f = int'(p_left);
        if (f > WS - int'(count)) f = WS - int'(count);
        if (!p_on) f = 0;
        n_fill = ($clog2(S+1))'(f);
    end
    assign pop_n = n_fill;
    assign st_issued = n_ret;
    always_comb begin st_reads = '0; for (int b = 0; b < NB; b++) st_reads = st_reads + ($clog2(NB+1))'(breq[b]); end

    logic [LANES-1:0] picked, ldone;
    logic issued_q;                                        // some lane added last cycle
    assign pass_idle = p_on && p_left == '0 && count == '0 && !issued_q && &ldone;
    // smallest tick that can still reach a lane: the oldest slot whose column has not landed, else the
    // last event taken into the window while more of the pass is to come (the stream is tick-ordered)
    logic [TIME-1:0] t_last;
    logic [TC-1:0] t_safe;
    always_comb begin
        logic found;
        found = 1'b0;
        t_safe = p_left != '0 ? TC'(t_last) : TC'(T);
        for (int k = 0; k < WS; k++) begin
            logic [WP-1:0] j;
            j = head + WP'(k);
            if (!found && k < int'(count) && !shd[j]) begin found = 1'b1; t_safe = TC'(stk[j]); end
        end
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            p_on <= 0; p_left <= '0; set <= 0; t_last <= '0; rbuf <= 0; slot_q <= '0; first_q <= 0; bank_q <= 0;
            sv <= '0; shd <= '0; srd <= '0; rd_v <= '0; head <= '0; tail <= '0; count <= '0; issued_q <= 0;
        end else begin
            for (int b = 0; b < NB; b++)
                if (rd_v[b]) scol[rd_slot[b]] <= bdata[b*LANES*W +: LANES*W];
            for (int j = 0; j < WS; j++) if (land[j]) begin shd[j] <= 1'b1; srd[j] <= 1'b0; end
            rd_v <= breq;
            for (int b = 0; b < NB; b++) if (breq[b]) begin rd_slot[b] <= req_slot[b]; srd[req_slot[b]] <= 1'b1; end
            for (int k = 0; k < S; k++)
                if (k < int'(n_ret)) begin sv[WP'(head + WP'(k))] <= 1'b0; shd[WP'(head + WP'(k))] <= 1'b0; end
            for (int k = 0; k < S; k++)
                if (k < int'(n_fill)) begin
                    sv[WP'(tail + WP'(k))] <= 1'b1; shd[WP'(tail + WP'(k))] <= 1'b0; srd[WP'(tail + WP'(k))] <= 1'b0;
                    sidx[WP'(tail + WP'(k))] <= evh[k*EW +: CIDX];
                    stk[WP'(tail + WP'(k))] <= evh[k*EW + CIDX +: TIME];
                end
            head <= head + WP'(n_ret);
            tail <= tail + WP'(n_fill);
            count <= count + (WP+1)'(n_fill) - (WP+1)'(n_ret);
            p_left <= p_left - (CIDX+1)'(n_fill);
            issued_q <= |picked;
            for (int k = 0; k < S; k++)
                if (k < int'(n_fill)) t_last <= evh[k*EW + CIDX +: TIME];
            if (pass_start) begin
                p_on <= 1; p_left <= pass_len; t_last <= '0;
                set <= pass_set; rbuf <= pass_wbuf; slot_q <= pass_slot; first_q <= pass_first; bank_q <= pass_bank;
            end else if (pass_end) p_on <= 0;
        end
    end

    // ---- membrane write-back: read the row, add every lane's pass result, write (2 cycles) -----
    logic ramp_set, ramp_first, ramp_bank;
    logic [SLOT-1:0] ramp_slot;
    logic [1:0] ramp_t;
    wire [LANES*ACC-1:0] row, orow;
    logic [LANES*ACC-1:0] new_row;
    wire ramp_write = ramp_busy && ramp_t == 2'd1;
    ttfs_mem_banks #(.OB(OB), .WIDTH(LANES*ACC)) membranes (
        .clk(clk), .write(ramp_write), .bank(ramp_bank), .addr(ramp_slot), .wdata(new_row), .rdata(row),
        .obank(cmp_bank), .oaddr(cmp_slot), .odata(orow));
    always_ff @(posedge clk) begin
        if (reset) begin
            ramp_busy <= 0; ramp_set <= 0; ramp_first <= 0; ramp_bank <= 0; ramp_slot <= '0; ramp_t <= '0;
        end else if (pass_end) begin
            ramp_busy <= 1; ramp_set <= set; ramp_slot <= slot_q; ramp_first <= first_q; ramp_bank <= bank_q;
            ramp_t <= '0;
        end else if (ramp_busy) begin
            if (ramp_write) ramp_busy <= 0;
            ramp_t <= ramp_t + 1'b1;
        end
    end

    // ---- output compaction (as ttfs_sp) --------------------------------------------------------
    logic [LANES*LR-1:0] cidx;
    logic [LR:0] ccount;
    logic cmp_q;
    ttfs_compact #(.LANES(LANES)) compact (.mask(spike_mask), .packed_idx(cidx), .count(ccount));
    always_ff @(posedge clk) begin
        if (reset) begin
            cmp_q <= 0; out_valid <= 0; out_idx <= '0; out_count <= '0;
        end else begin
            cmp_q <= compare;
            out_valid <= cmp_q;
            if (cmp_q) begin out_idx <= cidx; out_count <= ccount; end
        end
    end

    // ---- shared threshold table ------------------------------------------------------------------
    wire [LANES*TW-1:0] th_row;
    wire [LANES-1:0] th_dn;
    ttfs_thr #(.LANES(LANES), .ACC(ACC), .TW(TW), .T(T)) thresholds (
        .clk, .write(threshold_write), .wt(th_t), .wdata(threshold_data), .wdown(threshold_down),
        .rt(cmp_t), .rdata(th_row), .down(th_dn));

    // ---- lanes ---------------------------------------------------------------------------------
    logic [WS*TIME-1:0] stk_all;
    always_comb for (int j = 0; j < WS; j++) stk_all[j*TIME +: TIME] = stk[j];
    if (LANES_IMPL != 0) begin : g_lanes
        for (genvar l = 0; l < LANES; l = l+1) begin : lanes
            logic [WS-1:0] nz_l;
            logic [WS*W-1:0] w_l;
            always_comb for (int j = 0; j < WS; j++) begin
                nz_l[j] = lcol[j][l*W +: W] != '0;
                w_l[j*W +: W] = scol[j][l*W +: W];
            end
            ttfs_iv_lane #(.W(W), .B(B), .ACC(ACC), .T(T), .WS(WS), .TW(TW), .LS(LS), .GATE(GATE)) lane (
                .clk, .reset, .active(active[l]), .land, .land_nz(nz_l), .head, .sel_w(w_l), .sel_t(stk_all),
                .t_safe, .pend(pend_l[l]), .picked(picked[l]), .done(ldone[l]), .pass_start, .pass_end,
                .ramp_first, .old_v(row[l*ACC +: ACC]),
                .new_v(new_row[l*ACC +: ACC]), .cmp_load, .compare, .cmp_t, .cmp_v(orow[l*ACC +: ACC]),
                .th_cur(th_row[l*TW +: TW]), .down(th_dn[l]),
                .result(results[l*ACC +: ACC]), .spike(spike_mask[l]), .do_add(do_add[l]));
        end
    end else begin : g_nolanes
        for (genvar l = 0; l < LANES; l = l+1) begin : nl
            assign pend_l[l] = '0;
        end
        assign picked = '0; assign ldone = '1;
        assign new_row = '0; assign results = '0; assign spike_mask = '0; assign do_add = '0;
    end
    // synthesis translate_off
    always_ff @(posedge clk) if (!reset) begin
        assert (!(pass_end && !pass_idle)) else $fatal(1, "pass_end before the pass is done");
        assert (!(pass_end && ramp_busy)) else $fatal(1, "ramp unit busy");
        assert (!(cmp_load && compare)) else $fatal(1, "load and compare together");
        assert (!(pass_start && p_on && !pass_end)) else $fatal(1, "pass_start while a pass is open");
        assert (int'(ev_count) + int'(ev_push_n) - int'(pop_n) <= EDEPTH) else $fatal(1, "event FIFO overflow");
    end
    // synthesis translate_on
endmodule

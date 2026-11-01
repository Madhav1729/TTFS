// Storage and datapath blocks used by the TTFS tile.

// Event FIFO with multi-push.
module ttfs_evq #(parameter integer DEPTH = 32, WIDTH = 11, PUSH = 5, S = 4, PTR = $clog2(DEPTH)) (
    input  logic clk, reset,
    input  logic [$clog2(PUSH+1)-1:0] push_n,
    input  logic [PUSH*WIDTH-1:0] push_data,
    input  logic [$clog2(S+1)-1:0] pop_n,
    output logic [S*WIDTH-1:0] head,
    output logic [PTR:0] count
);
    logic [WIDTH-1:0] q [0:DEPTH-1];
    logic [PTR-1:0] rd, wr;
    always_comb for (int k = 0; k < S; k++) head[k*WIDTH +: WIDTH] = q[rd + PTR'(k)];
    always_ff @(posedge clk) begin
        if (reset) begin rd <= '0; wr <= '0; count <= '0; end
        else begin
            for (int k = 0; k < PUSH; k++)
                if (k < int'(push_n)) q[wr + PTR'(k)] <= push_data[k*WIDTH +: WIDTH];
            wr <= wr + PTR'(push_n);
            rd <= rd + PTR'(pop_n);
            count <= count + (PTR+1)'(push_n) - (PTR+1)'(pop_n);
        end
    end
endmodule

// Weight-column banks (SRAM-like, priced with CACTI).
module ttfs_wbanks #(parameter integer CH = 256, NB = 4, WIDTH = 256, AW = $clog2(CH), BW = $clog2(NB),
                     RW = AW - BW) (
    input  logic clk,
    input  logic write, wbuf,
    input  logic [AW-1:0] waddr,
    input  logic [WIDTH-1:0] wdata,
    input  logic rbuf,
    input  logic [NB-1:0] read,
    input  logic [NB*RW-1:0] raddr,
    output logic [NB*WIDTH-1:0] rdata                 // registered (SRAM-like) read, one per bank
);
    for (genvar b = 0; b < NB; b++) begin : bank
        logic [WIDTH-1:0] mem [0:2*(CH/NB)-1];
        always_ff @(posedge clk) begin
            if (write && waddr[BW-1:0] == BW'(b)) mem[{wbuf, waddr[AW-1:BW]}] <= wdata;
            if (read[b]) rdata[b*WIDTH +: WIDTH] <= mem[{rbuf, raddr[b*RW +: RW]}];
        end
    end
endmodule

// Expands a compressed weight column (lane mask + packed non-zeros).
module ttfs_unpack #(parameter integer LANES = 32, W = 4, R = $clog2(LANES)) (
    input  logic [LANES-1:0] mask,
    input  logic [LANES*W-1:0] packed_w,             // non-zero weights, lowest lane first
    output logic [LANES*W-1:0] dense
);
    always_comb begin
        logic [R:0] rank;
        rank = '0;
        dense = '0;
        for (int l = 0; l < LANES; l++) begin
            if (mask[l]) dense[l*W +: W] = packed_w[int'(rank)*W +: W];
            rank = rank + (R+1)'(mask[l]);
        end
    end
endmodule

// Double-buffered membrane rows (SRAM-like, priced with CACTI).
module ttfs_mem_banks #(parameter integer OB = 32, WIDTH = 1024, SLOT = $clog2(OB)) (
    input  logic clk,
    input  logic write, bank,
    input  logic [SLOT-1:0] addr,
    input  logic [WIDTH-1:0] wdata,
    output logic [WIDTH-1:0] rdata,
    input  logic obank,
    input  logic [SLOT-1:0] oaddr,
    output logic [WIDTH-1:0] odata
);
    logic [WIDTH-1:0] rows [0:2*OB-1];
    always_ff @(posedge clk) if (write) rows[{bank, addr}] <= wdata;
    assign rdata = rows[{bank, addr}];
    assign odata = rows[{obank, oaddr}];
endmodule

// Compacts an output tick's spike mask into lane indices.
module ttfs_compact #(parameter integer LANES = 32, R = $clog2(LANES)) (
    input  logic [LANES-1:0] mask,
    output logic [LANES*R-1:0] packed_idx,
    output logic [R:0] count
);
    logic [LANES*R-1:0] stage [0:R];
    logic [LANES*R-1:0] shift [0:R];
    logic [LANES-1:0] live [0:R];
    always_comb begin
        logic [R:0] zeros;
        zeros = '0;
        for (int i = 0; i < LANES; i++) begin
            stage[0][i*R +: R] = R'(i);
            shift[0][i*R +: R] = R'(zeros);
            live[0][i] = mask[i];
            zeros = zeros + (R+1)'(!mask[i]);
        end
        for (int k = 0; k < R; k++) begin
            stage[k+1] = '0; shift[k+1] = '0; live[k+1] = '0;
            for (int i = 0; i < LANES; i++)
                if (live[k][i]) begin
                    if (shift[k][i*R + k] && i >= (1 << k)) begin
                        stage[k+1][(i - (1 << k))*R +: R] = stage[k][i*R +: R];
                        shift[k+1][(i - (1 << k))*R +: R] = shift[k][i*R +: R];
                        live[k+1][i - (1 << k)] = 1;
                    end else begin
                        stage[k+1][i*R +: R] = stage[k][i*R +: R];
                        shift[k+1][i*R +: R] = shift[k][i*R +: R];
                        live[k+1][i] = 1;
                    end
                end
        end
        packed_idx = stage[R];
        count = (R+1)'(LANES) - zeros;
    end
endmodule

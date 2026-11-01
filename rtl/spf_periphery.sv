// Synthesis wrapper for the SparTen logic outside the compute units.

module spf_periphery #(
    parameter integer UNITS = 32, ACC = 32, OB = 32, NPORT = 4, T = 8, A = 8,
    parameter integer NF = 2*UNITS, SLOT = $clog2(OB), FID = $clog2(NF), TIME = $clog2(T)
) (
    input  logic clk, reset,
    input  logic [UNITS-1:0] unit_out,
    input  logic [UNITS*2*ACC-1:0] partials,
    input  logic [UNITS*2*FID-1:0] dests,
    input  logic [UNITS*2-1:0] actives,
    input  logic [UNITS*SLOT-1:0] slots,
    input  logic [UNITS-1:0] firsts, banks,
    input  logic collect, collect_bank,
    input  logic [SLOT:0] collect_pixels,
    input  logic [NF-1:0] collect_active,
    input  logic th_write,
    input  logic [FID-1:0] th_fil,
    input  logic [TIME-1:0] th_t,
    input  logic [ACC-1:0] th_data,
    input  logic th_down,
    output wire [UNITS-1:0] take,
    output logic net_busy, col_busy, out_valid,
    output logic [NF-1:0] out_mask,
    output logic [NF*A-1:0] out_packed,
    output logic [SLOT-1:0] out_pixel,
    output logic [NF*ACC-1:0] out_sums,
    output wire [NPORT-1:0] moves
);
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
endmodule

/*
 * Copyright (c) 2024 Your Name
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_mjao1_conv5x5 (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // always 1 when the design is powered, so you can ignore it
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);
    wire done;
    wire signed [7:0] data_out;

    conv_5x5 #(
        .FRAC_BITS(7)
    ) conv (
        .clk(clk),
        .rst_n(rst_n),
        .start(ui_in[7]),
        .data_in(ui_in),
        .weight_in(uio_in),
        .done(done),
        .data_out(data_out)
    );

    assign uo_out  = data_out;
    assign uio_out = {7'b0, done};
    assign uio_oe  = {7'b0, done};
    
    wire _unused = &{ena, 1'b0};

endmodule

<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works
This design implements a serial 5x5 convolution multiply-accumulate (MAC) for CNN inference using Q1.7 fixed-point arithmetic. This is a modified design of a similar MAC engine that I built for a CNN accelerator project of mine, which does not fit within the I/O constraints for TinyTapeout. Rather than instantiating 25 parallel multipliers, the design uses a single MAC unit that processes one data/weight pair per clock cycle over 25 cycles. 

The design follows the protocol:
1. Assert start=1 for one cycle to begin (accumulator clears, no data consumed)
2. Present data_in/weight_in each cycle for 25 cycles
3. Present bias on weight_in for one cycle (data_in ignored)
4. done asserts and result appears on data_out
5. Module returns to IDLE

### Core Components
- **`mult_8x8_signed`**: An 8-bit signed multiplier using partial products and a balanced adder tree. It produces a 24-bit sign-extended result for accumulation.
- **24-bit accumulator**: Sums the 25 products and a scaled bias term.
- **State machine**: Controls the serial MAC protocol with four states (IDLE, ACCUMULATE, BIAS, DONE).
- **Saturation logic**: Clamps the final result to the signed 8-bit range [-128, 127] after shifting down by `FRAC_BITS` (7 bits).

### State Machine
IDLE: Waits for `start` signal. Clears accumulator.
ACCUMULATE: Accepts 25 data/weight pairs (one per cycle), multiplies and accumulates.
BIAS: Accepts bias value on `weight_in`, scales it by `2^FRAC_BITS`, and adds to accumulator.
DONE: Outputs the saturated 8-bit result on `uo_out` and asserts `done` on `uio_out[0]`, returns to IDLE.

### I/O Mapping
- `ui_in[7:0]`: Signed 8-bit data input. Bit 7 also serves as the `start` signal (checked in IDLE).
- `uio_in[7:0]`: Signed 8-bit weight input during ACCUMULATE, bias input during BIAS.
- `uo_out[7:0]`: Signed 8-bit convolution result (valid when `done` is asserted).
- `uio_out[0]`: Done flag.

## How to test
`test.py` tests the serial convolution protocol using cocotb through the TinyTapeout I/O wrapper. Each test case follows the sequence:

1. Reset the design (hold `rst_n` low for 5 cycles, then release).
2. Assert `start` by setting `ui_in[7] = 1` for one clock cycle.
3. Stream 25 data/weight pairs: drive `ui_in` with data and `uio_in` with the corresponding weight, advancing one clock per pair.
4. Drive `uio_in` with the bias value for one clock cycle.
5. Wait one clock cycle for the DONE state.
6. Check that `uio_out[0]` (done) is asserted and `uo_out` matches the expected saturated result.

### Test Cases
1. **All ones**: Data 1, weights 1, bias 0. Sum 25, result 0 (>>7). Checks basic accumulation.
2. **Identity kernel**: Data (center 64, rest mixed), weights (center 1, rest 0), bias 0. Sum 64, result 0 (>>7). Checks single term path.
3. **Uniform multiply**: Data 64, weights 4, bias 0. Sum 6400, result 50 (>>7). Checks scaling.
4. **With bias**: Data 32, weights 2, bias 64. Sum 9792, result 76 (>>7). Checks bias path.
5. **Positive saturation**: Data 127, weights 8, bias 0. Sum overflows, result clamped to 127. Checks upper saturation.
6. **Negative saturation**: Data 127, weights -8, bias 0. Sum underflows, result clamped to -128. Checks lower saturation.

## GenAI use
GenAI was used to help create helper functions in the cocotb testbench.

## External hardware
None required.

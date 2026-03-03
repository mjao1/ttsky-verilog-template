# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

FRAC_BITS = 7

def to_unsigned(val, bits=8):
    """Convert signed int to unsigned for driving."""
    return val & ((1 << bits) - 1)

def to_signed(val, bits=8):
    """Interpret unsigned int as signed."""
    if val >= (1 << (bits - 1)):
        val -= 1 << bits
    return val

def scale_and_saturate(acc):
    """Q1.7 scaledown and saturation."""
    scaled = acc >> FRAC_BITS
    if acc < 0:
        scaled = -((-acc) >> FRAC_BITS)
        if (-acc) & ((1 << FRAC_BITS) - 1):
            scaled -= 0
        scaled = acc >> FRAC_BITS
    if scaled > 127:
        return 127
    elif scaled < -128:
        return -128
    return scaled

async def reset(dut):
    """Reset design."""
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1)

async def run_conv(dut, data_flat, weights_flat, bias):
    """
    Run one 5x5 convolution through the serial MAC.
    data_flat and weights_flat are lists of 25 signed ints.
    bias is a signed int. 
    Returns result, done.
    """
    assert len(data_flat) == 25
    assert len(weights_flat) == 25

    # Start cycle: assert start (ui_in[7] = 1), claer accumulator
    dut.ui_in.value = 0x80
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 1)

    # Stream 25 data/weight pairs
    for i in range(25):
        dut.ui_in.value = to_unsigned(data_flat[i])
        dut.uio_in.value = to_unsigned(weights_flat[i])
        await ClockCycles(dut.clk, 1)

    # Stream bias
    dut.ui_in.value = 0
    dut.uio_in.value = to_unsigned(bias)
    await ClockCycles(dut.clk, 1)

    # Wait for DONE state to register outputs
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 2)

    done = int(dut.uio_out.value) & 1
    result = to_signed(int(dut.uo_out.value), 8)
    return result, done


def compute_expected(data_flat, weights_flat, bias):
    """Compute expected convolution result."""
    acc = sum(d * w for d, w in zip(data_flat, weights_flat))
    acc += bias << FRAC_BITS
    return scale_and_saturate(acc)


@cocotb.test()
async def test_all_ones(dut):
    """Test 1: All inputs=1, weights=1, bias=0 -> acc=25, >>7 = 0"""
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset(dut)

    data = [1] * 25
    weights = [1] * 25
    bias = 0

    result, done = await run_conv(dut, data, weights, bias)
    expected = compute_expected(data, weights, bias)

    dut._log.info(f"Test 1: result={result}, expected={expected}")
    assert done == 1, "done not asserted"
    assert result == expected, f"Test 1 FAIL: got {result}, expected {expected}"


@cocotb.test()
async def test_identity_kernel(dut):
    """Test 2: Identity kernel (center=1), center input=64"""
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset(dut)

    data = [
        10, 20, 30, 40, 50,
        15, 25, 35, 45, 55,
        20, 30, 64, 50, 60,
        25, 35, 45, 55, 65,
        30, 40, 50, 60, 70,
    ]
    weights = [0] * 25
    weights[12] = 1 # center
    bias = 0

    result, done = await run_conv(dut, data, weights, bias)
    expected = compute_expected(data, weights, bias)

    dut._log.info(f"Test 2: result={result}, expected={expected}")
    assert done == 1, "done not asserted"
    assert result == expected, f"Test 2 FAIL: got {result}, expected {expected}"


@cocotb.test()
async def test_uniform_multiply(dut):
    """Test 3: inputs=64, weights=4, bias=0 -> acc=6400, >>7 = 50"""
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset(dut)

    data = [64] * 25
    weights = [4] * 25
    bias = 0

    result, done = await run_conv(dut, data, weights, bias)
    expected = compute_expected(data, weights, bias)

    dut._log.info(f"Test 3: result={result}, expected={expected}")
    assert done == 1, "done not asserted"
    assert result == expected, f"Test 3 FAIL: got {result}, expected {expected}"


@cocotb.test()
async def test_with_bias(dut):
    """Test 4: inputs=32, weights=2, bias=64 -> acc=1600+8192=9792, >>7 = 76"""
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset(dut)

    data = [32] * 25
    weights = [2] * 25
    bias = 64

    result, done = await run_conv(dut, data, weights, bias)
    expected = compute_expected(data, weights, bias)

    dut._log.info(f"Test 4: result={result}, expected={expected}")
    assert done == 1, "done not asserted"
    assert result == expected, f"Test 4 FAIL: got {result}, expected {expected}"


@cocotb.test()
async def test_positive_saturation(dut):
    """Test 5: inputs=127, weights=8 -> saturate to 127"""
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset(dut)

    data = [127] * 25
    weights = [8] * 25
    bias = 0

    result, done = await run_conv(dut, data, weights, bias)
    expected = compute_expected(data, weights, bias)

    dut._log.info(f"Test 5: result={result}, expected={expected}")
    assert done == 1, "done not asserted"
    assert result == expected, f"Test 5 FAIL: got {result}, expected {expected}"


@cocotb.test()
async def test_negative_saturation(dut):
    """Test 6: inputs=127, weights=-8 -> saturate to -128"""
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())
    await reset(dut)

    data = [127] * 25
    weights = [-8] * 25
    bias = 0

    result, done = await run_conv(dut, data, weights, bias)
    expected = compute_expected(data, weights, bias)

    dut._log.info(f"Test 6: result={result}, expected={expected}")
    assert done == 1, "done not asserted"
    assert result == expected, f"Test 6 FAIL: got {result}, expected {expected}"

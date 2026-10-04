"""std.tile: the elementwise and row/column vocabulary over the four
higher-order primitives, checked against Python oracles in f64, f32
(per-op rounding through narrow mode) and f16, and natively for f64.
"""
from __future__ import annotations

import math
import struct

import pytest

from metaxu.compiler.frozen_borrow_checker import TypeCheckError
from metaxu.compiler.pipeline import build_context_from_source, run_pipeline_ctx
from metaxu.compiler.tests.test_codegen_llvm import (
    assert_native_matches_interp, count_placeholders, interp_run,
    llvm_from_source, needs_clang)

ROWS = [[0.5, 1.5, -2.0, 0.25], [3.25, 0.0, 1.0, -0.75], [2.0, 2.0, -1.5, 0.125]]
VALS = [x for r in ROWS for x in r]


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def _f16(x):
    try:
        return struct.unpack("e", struct.pack("e", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)   # IEEE overflow, like (half)x


def _program(conv: str) -> str:
    pushes = " ".join(f"v.push({x!r});" for x in VALS)
    return f"""
from std.tile import exp, log, sqrt, neg, sub, div, maximum, minimum,
    row_max, row_sum, col_max, col_sum, add_rows, sub_rows, mul_rows, div_rows,
    add_cols, sub_cols, mul_cols, div_cols, softmax_rows, neg_huge;
fn main() -> int {{
    let @mut v = Vec.new();
    {pushes}
    let t = {conv}(Tile.from_vec(v, 3, 4));
    print(softmax_rows(t));
    print(row_max(t));
    print(row_sum(t));
    print(col_max(t));
    print(col_sum(t));
    print(sub_rows(t, row_max(t)));
    print(div_cols(mul_rows(t, row_sum(t)), col_sum(t)));
    print(add_rows(t, row_max(t)));
    print(add_cols(sub_cols(t, col_max(t)), col_sum(t)));
    print(exp(neg(t)));
    print(sqrt(maximum(t, {conv}(Tile.filled(3, 4, 0.0)))));
    print(log(div(exp(t), sub(t, minimum(t, {conv}(Tile.filled(3, 4, -4.0)))))));
    print(mul_cols(t, col_sum(t)));
    0
}}
"""


class _Oracle:
    """The same operations in Python under a rounding function `r`
    (identity for f64), each op rounded once, folds left to right."""

    def __init__(self, r):
        self.r = r
        self.rows = [[r(x) for x in row] for row in ROWS]

    def tile(self, rows):
        return "tile[3x4](" + "; ".join(", ".join(repr(x) for x in row) for row in rows) + ")"

    def col(self, vals):
        return "tile[3x1](" + "; ".join(repr(x) for x in vals) + ")"

    def row(self, vals):
        return "tile[1x4](" + ", ".join(repr(x) for x in vals) + ")"

    def row_max(self, rows=None):
        rows = rows or self.rows
        out = []
        for row in rows:
            m = self.r(-3.4028234663852886e38)
            for x in row:
                m = x if x > m else m
            out.append(m)
        return out

    def row_sum(self, rows=None):
        rows = rows or self.rows
        out = []
        for row in rows:
            s = self.r(0.0)
            for x in row:
                s = self.r(s + x)
            out.append(s)
        return out

    def col_max(self):
        out = []
        for j in range(4):
            m = self.r(-3.4028234663852886e38)
            for row in self.rows:
                m = row[j] if row[j] > m else m
            out.append(m)
        return out

    def col_sum(self):
        out = []
        for j in range(4):
            s = self.r(0.0)
            for row in self.rows:
                s = self.r(s + row[j])
            out.append(s)
        return out

    def map(self, f, rows=None):
        return [[self.r(f(x)) for x in row] for row in (rows or self.rows)]

    def zip(self, f, a, b):
        return [[self.r(f(x, y)) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]

    def brows(self, f, rows, v):
        return [[self.r(f(x, s)) for x in row] for row, s in zip(rows, v)]

    def bcols(self, f, rows, v):
        return [[self.r(f(x, v[j])) for j, x in enumerate(row)] for row in rows]

    def expected(self):
        r = self.r
        t = self.rows
        rm, rs, cm, cs = self.row_max(), self.row_sum(), self.col_max(), self.col_sum()
        e = self.map(math.exp, self.brows(lambda x, s: x - s, t, rm))
        soft = self.brows(lambda x, s: x / s, e, self.row_sum(e))
        zeros = [[r(0.0)] * 4 for _ in range(3)]
        m4 = [[r(-4.0)] * 4 for _ in range(3)]
        mx = self.zip(lambda a, b: b if b > a else a, t, zeros)
        mn = self.zip(lambda a, b: b if b < a else a, t, m4)
        return [
            self.tile(soft),
            self.col(rm), self.col(rs), self.row(cm), self.row(cs),
            self.tile(self.brows(lambda x, s: x - s, t, rm)),
            self.tile(self.bcols(lambda x, s: x / s, self.brows(lambda x, s: x * s, t, rs), cs)),
            self.tile(self.brows(lambda x, s: x + s, t, rm)),
            self.tile(self.bcols(lambda x, s: x + s, self.bcols(lambda x, s: x - s, t, cm), cs)),
            self.tile(self.map(math.exp, self.map(lambda x: 0.0 - x))),
            self.tile(self.map(math.sqrt, mx)),
            self.tile(self.map(math.log, self.zip(lambda a, b: a / b, self.map(math.exp),
                                                   self.zip(lambda a, b: a - b, t, mn)))),
            self.tile(self.bcols(lambda x, s: x * s, t, cs)),
        ]


@pytest.mark.parametrize("conv,rnd", [("Tile.to_f64", lambda x: float(x)),
                                      ("Tile.to_f32", _f32), ("Tile.to_f16", _f16)],
                         ids=["f64", "f32", "f16"])
def test_interp_matches_python_oracle(conv, rnd):
    result, out = interp_run(_program(conv))
    assert result == 0
    assert out.splitlines() == _Oracle(rnd).expected()


def test_f32_differs_from_rounding_once():
    # The narrow-mode results are not what rounding the f64 answers would
    # give: the oracle rounds per op, and so does the interpreter.
    o64 = _Oracle(lambda x: float(x)).expected()
    o32 = _Oracle(_f32).expected()
    rounded_once = _Oracle(lambda x: float(x))
    assert o32[0] != o64[0]
    import re
    nums64 = [float(x) for x in re.findall(r"-?\d+\.\d+(?:e[-+]\d+)?", o64[0])]
    nums32 = [float(x) for x in re.findall(r"-?\d+\.\d+(?:e[-+]\d+)?", o32[0])]
    assert any(_f32(a) != b for a, b in zip(nums64, nums32))
    del rounded_once


def test_neg_huge_is_the_most_negative_f32():
    _res, out = interp_run("from std.tile import neg_huge;\n"
                           "fn main() -> int { print(neg_huge()); 0 }")
    assert float(out.strip()) == -3.4028234663852886e38
    assert _f32(float(out.strip())) == float(out.strip())


@pytest.mark.parametrize("expr,message", [
    ("sub_rows(Tile.filled(2, 3, 1.0), Tile.filled(1, 3, 1.0))",
     "Tile.broadcast_rows: shape mismatch: expected a 2x1 column for a 2x3 tile, got 1x3"),
    ("sub(Tile.filled(2, 3, 1.0), Tile.filled(3, 2, 1.0))",
     "Tile.zip: shape mismatch: 2x3 vs 3x2"),
])
def test_library_shape_misuse_is_a_loud_runtime_error(expr, message):
    # The helpers take shape-generic parameters, so the static checker has
    # nothing to compare inside them; the mismatch is caught at the
    # primitive at run time, loudly and catchably.
    src = ("from std.tile import sub_rows, sub;\n"
           "fn main() -> int {\n"
           f"    let r = try {{ let t = {expr}; 0 }} catch e {{ print(e); 1 }};\n"
           "    r\n}\n")
    result, out = interp_run(src)
    assert result == 1
    assert out.strip() == message


@pytest.mark.parametrize("conv", ["Tile.to_f64", "Tile.to_f32", "Tile.to_f16"],
                         ids=["f64", "f32", "f16"])
def test_library_lowers_natively(conv):
    ir = llvm_from_source(_program(conv))
    assert count_placeholders(ir) == 0
    if conv != "Tile.to_f64":
        # every lambda of the library is a narrow lambda here
        assert "narrow lambda:" in ir
        assert ("to float" if conv == "Tile.to_f32" else "to half") in ir


@needs_clang
@pytest.mark.parametrize("conv", ["Tile.to_f64", "Tile.to_f32", "Tile.to_f16"],
                         ids=["f64", "f32", "f16"])
def test_native_library_matches_interp(conv, tmp_path):
    # The same oracle-checked program, natively: the narrow lambdas round
    # per op exactly as the interpreter's narrow mode does.
    assert_native_matches_interp(_program(conv), tmp_path)

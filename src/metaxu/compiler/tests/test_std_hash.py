"""std.sha256 and std.hex (Metaxu) against hashlib and binascii, the oracle.

One generated program hashes messages of every interesting length (the
padding boundaries at 55, 56, 63, 64 bytes and a longer one), raw byte
patterns and UTF-8 text, prints each digest and a few hex round trips;
Python computes the same; the transcripts must match, on the
interpreter and then natively.
"""
from __future__ import annotations

import hashlib
import random
import shutil

import pytest

from metaxu.compiler.tests.test_codegen_llvm import interp_run, llvm_from_source

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="clang is not installed")

TEXTS = ["", "abc", "message digest", "The quick brown fox jumps over the lazy dog",
         "héllo wörld", "a" * 55, "b" * 56, "c" * 63, "d" * 64, "e" * 65, "f" * 200]
HEX_INPUTS = ["", "00", "ff", "0aFF10", "abc", "zz", "deadbeef"]


def _byte_patterns() -> list[list[int]]:
    rng = random.Random(3)
    pats = [[0], [255], list(range(256)), [0] * 64, [255] * 119]
    for n in (1, 7, 31, 100):
        pats.append([rng.randint(0, 255) for _ in range(n)])
    return pats


def program() -> str:
    lines = ["from std.sha256 import sha256_hex, sha256_string;",
             "from std.hex import to_hex, from_hex;",
             "fn bytes_of(parts: string) -> Vec {",
             "    let @mut out = Vec.new();",
             "    if parts == \"\" { () } else {",
             "        let fields = parts.split(\",\");",
             "        let @mut i = 0;",
             "        while i < len(fields) {",
             "            match parse_byte(fields[i]) { Some(b) => out.push(b), None => () };",
             "            i = i + 1",
             "        }",
             "    };",
             "    out",
             "}",
             "fn parse_byte(s: string) -> Option {",
             "    let @mut v = 0;",
             "    let @mut i = 0;",
             "    while i < len(s) { v = v * 10 + (\"0123456789\".find(s[i])); i = i + 1 };",
             "    Some(v)",
             "}",
             "fn main() -> int {"]
    for t in TEXTS:
        lines.append(f'    print(sha256_string("{t}"));')
    for pat in _byte_patterns():
        joined = ",".join(str(b) for b in pat)
        lines.append(f'    print(sha256_hex(bytes_of("{joined}")));')
        lines.append(f'    print(to_hex(bytes_of("{joined}")));')
    for h in HEX_INPUTS:
        lines.append(f'    match from_hex("{h}") {{ Some(b) => print(to_hex(b)), None => print("bad") }};')
    lines.append("    0")
    lines.append("}")
    return "\n".join(lines) + "\n"


def oracle() -> list[str]:
    out: list[str] = []
    for t in TEXTS:
        out.append(hashlib.sha256(t.encode("utf-8")).hexdigest())
    for pat in _byte_patterns():
        out.append(hashlib.sha256(bytes(pat)).hexdigest())
        out.append(bytes(pat).hex())
    for h in HEX_INPUTS:
        try:
            out.append(bytes.fromhex(h).hex() if len(h) % 2 == 0 else "bad")
        except ValueError:
            out.append("bad")
    return out


@pytest.fixture(scope="module")
def transcript() -> list[str]:
    _result, out = interp_run(program())
    return out.rstrip("\n").split("\n")


def test_interpreter_matches_hashlib(transcript):
    assert transcript == oracle()


def test_the_well_known_vectors(transcript):
    assert transcript[0] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert transcript[1] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_native_is_placeholder_free():
    ir = llvm_from_source(program())
    assert "placeholder -- unsupported" not in ir


@needs_clang
def test_native_prints_what_the_interpreter_printed(tmp_path, transcript):
    from metaxu.compiler.llvm_run import compile_and_run
    exit_code, stdout = compile_and_run(llvm_from_source(program()), "main",
                                        workdir=str(tmp_path), timeout=300)
    assert exit_code == 0
    assert stdout.rstrip("\n").split("\n") == transcript

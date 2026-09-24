"""std.semver (Metaxu) against glade's semver.py (Python), the oracle.

One generated Metaxu program prints an answer per case; the same cases
go through the Python module; the two transcripts must match line for
line. Cases cover version parsing (valid and malformed), ordering,
requirement parsing in the Cargo dialect, membership including the
prerelease rule, and the range algebra the solver relies on. The native
backend then has to print exactly what the interpreter printed.
"""
from __future__ import annotations

import random
import shutil

import pytest

from metaxu.compiler.tests.test_codegen_llvm import interp_run
from metaxu.glade.semver import Range, Version, VersionError, parse_requirement

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="clang is not installed")

# --- the cases ----------------------------------------------------------------

VERSIONS = [
    "0.0.0", "1.2.3", "10.20.30", "1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta",
    "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0", "2.0.0-rc.1",
    "1.2.3-rc.1+build.5", "1.2.3+build", "0.9.9", "1.0.0-0.3.7", "1.0.0-x.7.z.92",
    "1.0.0-x-y-z.--", " 1.2.3 ",
    # malformed
    "1.2", "1", "01.2.3", "1.02.3", "1.2.03", "1.0.0-", "1.0.0-a..b", "1.0.0+", "1.0.0+a..b",
    "1.0.0-al pha", "v1.2.3", "1.2.3.4", "", "a.b.c", "1.2.3-", "1.2.3-rc/1",
]

REQUIREMENTS = [
    "^1.2.3", "1.2.3", "^0.2.3", "^0.0.3", "^0.0", "^0", "^1", "^1.2", "~1.2.3", "~1.2", "~1",
    "1.2.*", "1.*", "*", "", "=1.2.3", "=1.2", "=1", ">=1.2, <2", ">1.2", "<=1.2", "<1.2",
    ">1.2.3", ">=1.0.0-rc.1", "^1.0", ">=0.1.0, <0.1.5", ">=1, <1", "^1, ^2", "^ 1.2",
    " >= 1.0 , < 3.0 ", "1.2.3-rc.1", "^2.0.0-rc.1", "<1.0.0-rc.1", ">=1.5, <3", "~0.0.1",
    # malformed
    "1.2.x", "1.*.3", "^", "~", ">=", "1..2", "^1.2.3.4", "1.2.3,", ",", "a", "^a", "1.2.3 1.2.4",
    "1.0.0-", "^1.0.0-a..b", "**",
]

CONTAINS_VERSIONS = [
    "0.0.0", "0.0.3", "0.0.4", "0.1.0", "0.1.4", "0.1.5", "0.2.3", "0.2.9", "0.3.0", "0.9.0",
    "1.0.0", "1.0.0-rc.1", "1.0.0-rc.2", "1.0.0-alpha", "1.1.0", "1.2.0", "1.2.2", "1.2.3",
    "1.2.3-rc.1", "1.2.9", "1.3.0", "1.5.0-beta", "1.9.9", "2.0.0", "2.0.0-rc.1", "2.0.0-rc.2",
    "2.5.0", "3.0.0",
]


def _random_cases(seed: int = 7):
    rng = random.Random(seed)
    pres = ["", "-alpha", "-beta.2", "-rc.1", "-0", "-1.a"]
    ops = ["^", "~", "", ">=", ">", "<", "<=", "="]
    versions, reqs = [], []
    for _ in range(60):
        v = f"{rng.randint(0, 3)}.{rng.randint(0, 4)}.{rng.randint(0, 5)}{rng.choice(pres)}"
        versions.append(v)
    for _ in range(60):
        parts = [str(rng.randint(0, 3))]
        depth = rng.randint(1, 3)
        if depth >= 2:
            parts.append(rng.choice([str(rng.randint(0, 4)), "*"]))
        if depth == 3 and parts[-1] != "*":
            parts.append(rng.choice([str(rng.randint(0, 5)), "*"]))
        req = rng.choice(ops) + ".".join(parts)
        if rng.random() < 0.3:
            req += f", <{rng.randint(1, 4)}"
        reqs.append(req)
    return versions, reqs


RVERS, RREQS = _random_cases()
ALL_VERSIONS = VERSIONS + RVERS
ALL_REQS = REQUIREMENTS + RREQS
CMP_PAIRS = [(a, b) for i, a in enumerate(VERSIONS[:19]) for b in VERSIONS[:19][i:]]
ALGEBRA_PAIRS = [(REQUIREMENTS[i], REQUIREMENTS[j]) for i in range(0, 34, 3) for j in range(1, 34, 5)]


# --- the oracle -------------------------------------------------------------------

def _bool(b: bool) -> str:
    return "1" if b else "0"      # both engines print a bool as its word


def oracle() -> list[str]:
    out: list[str] = []
    for s in ALL_VERSIONS:
        try:
            out.append(f"v {s!r} {Version.parse(s)}")
        except VersionError:
            out.append(f"v {s!r} invalid")
    for a, b in CMP_PAIRS:
        va, vb = Version.parse(a), Version.parse(b)
        out.append(f"c {a} {b} {-1 if va < vb else (1 if vb < va else 0)}")
    for s in ALL_REQS:
        try:
            out.append(f"r {s!r} {parse_requirement(s)}")
        except VersionError:
            out.append(f"r {s!r} invalid")
    for req in REQUIREMENTS[:35]:
        r = parse_requirement(req)
        for v in CONTAINS_VERSIONS:
            out.append(f"m {req!r} {v} {_bool(r.contains(Version.parse(v)))}")
    for a, b in ALGEBRA_PAIRS:
        ra, rb = parse_requirement(a), parse_requirement(b)
        out.append(f"a {a!r} {b!r} {ra.intersect(rb)} | {ra.union(rb)} | {ra.complement()} | "
                   f"{_bool(ra.is_subset_of(rb))} {_bool(ra.is_disjoint_from(rb))}")
    return out


# --- the Metaxu program -------------------------------------------------------------

def _lit(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def program() -> str:
    lines = [
        "from std.semver import parse_version, version_to_string, compare_version, parse_requirement,",
        "    range_to_string, range_contains, range_intersect, range_union, range_complement,",
        "    range_is_subset, range_is_disjoint;",
        "",
        "fn show_version(label: string, s: string) -> () {",
        "    match parse_version(s) {",
        '        None => print("v " + label + " invalid"),',
        '        Some(v) => print("v " + label + " " + version_to_string(v))',
        "    }",
        "}",
        "fn show_cmp(a: string, b: string) -> () {",
        "    match parse_version(a) { None => print(\"?\"), Some(va) =>",
        "    match parse_version(b) { None => print(\"?\"), Some(vb) =>",
        '        print("c " + a + " " + b + " " + compare_version(va, vb).to_string()) } }',
        "}",
        "fn show_req(label: string, s: string) -> () {",
        "    match parse_requirement(s) {",
        '        None => print("r " + label + " invalid"),',
        '        Some(r) => print("r " + label + " " + range_to_string(r))',
        "    }",
        "}",
        "fn flag(b: bool) -> string { if b { \"1\" } else { \"0\" } }",
        "fn show_contains(label: string, req: string, v: string) -> () {",
        "    match parse_requirement(req) { None => print(\"?\"), Some(r) =>",
        "    match parse_version(v) { None => print(\"?\"), Some(ver) =>",
        '        print("m " + label + " " + v + " " + flag(range_contains(r, ver))) } }',
        "}",
        "fn show_algebra(la: string, lb: string, a: string, b: string) -> () {",
        "    match parse_requirement(a) { None => print(\"?\"), Some(ra) =>",
        "    match parse_requirement(b) { None => print(\"?\"), Some(rb) =>",
        '        print("a " + la + " " + lb + " " + range_to_string(range_intersect(ra, rb))',
        '              + " | " + range_to_string(range_union(ra, rb))',
        '              + " | " + range_to_string(range_complement(ra))',
        '              + " | " + flag(range_is_subset(ra, rb)) + " " + flag(range_is_disjoint(ra, rb))) } }',
        "}",
        "fn main() -> int {",
    ]
    for s in ALL_VERSIONS:
        lines.append(f"    show_version({_lit(repr(s))}, {_lit(s)});")
    for a, b in CMP_PAIRS:
        lines.append(f"    show_cmp({_lit(a)}, {_lit(b)});")
    for s in ALL_REQS:
        lines.append(f"    show_req({_lit(repr(s))}, {_lit(s)});")
    for req in REQUIREMENTS[:35]:
        for v in CONTAINS_VERSIONS:
            lines.append(f"    show_contains({_lit(repr(req))}, {_lit(req)}, {_lit(v)});")
    for a, b in ALGEBRA_PAIRS:
        lines.append(f"    show_algebra({_lit(repr(a))}, {_lit(repr(b))}, {_lit(a)}, {_lit(b)});")
    lines += ["    0", "}", ""]
    return "\n".join(lines)


@pytest.fixture(scope="module")
def transcript():
    result, out = interp_run(program())
    assert result == 0
    return out.rstrip("\n").split("\n")


def test_case_counts_are_what_the_oracle_expects(transcript):
    assert len(transcript) == len(oracle())


def test_versions_parse_and_print_like_python(transcript):
    got = [l for l in transcript if l.startswith("v ")]
    assert got == [l for l in oracle() if l.startswith("v ")]


def test_ordering_matches_semver(transcript):
    got = [l for l in transcript if l.startswith("c ")]
    assert got == [l for l in oracle() if l.startswith("c ")]


def test_requirements_parse_and_print_like_python(transcript):
    got = [l for l in transcript if l.startswith("r ")]
    assert got == [l for l in oracle() if l.startswith("r ")]


def test_membership_including_the_prerelease_rule(transcript):
    got = [l for l in transcript if l.startswith("m ")]
    assert got == [l for l in oracle() if l.startswith("m ")]


def test_range_algebra(transcript):
    got = [l for l in transcript if l.startswith("a ")]
    assert got == [l for l in oracle() if l.startswith("a ")]


@needs_clang
def test_native_prints_what_the_interpreter_printed(tmp_path, transcript):
    # This module found two native-backend gaps (docs/glade_in_metaxu.md):
    # string indexing was not lowered, and a nested `Some(Some(n))` read its
    # inner payload through the module-wide Option cells, which conflict as
    # soon as one file puts ints, Vecs and structs into Option.  Both are
    # fixed; this differential is the pin.
    from metaxu.compiler.tests.test_codegen_llvm import llvm_from_source
    from metaxu.compiler.llvm_run import compile_and_run
    exit_code, stdout = compile_and_run(llvm_from_source(program()), "main",
                                        workdir=str(tmp_path), timeout=300)
    assert exit_code == 0
    assert stdout.rstrip("\n").split("\n") == transcript

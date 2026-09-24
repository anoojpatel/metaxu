"""std.solve (Metaxu) against glade's pubgrub.py (Python), the oracle.

Each scenario is a dependency graph. The Python solver and the Metaxu
port both solve it; the picks must be identical, and when there is no
solution the explanation must match sentence for sentence. The
scenarios are those of test_glade_solver.py plus a few with more
backtracking, and the fifty-version graph that checks the widening
step still keeps the derivation short.
"""
from __future__ import annotations

import shutil

import pytest

from metaxu.compiler.tests.test_codegen_llvm import interp_run
from metaxu.glade.pubgrub import NoSolution, Solver
from metaxu.glade.semver import Version, parse_requirement

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="clang is not installed")

Graph = dict[str, dict[str, dict[str, str]]]

SCENARIOS: list[tuple[str, Graph, dict[str, str], dict[str, str]]] = [
    # name, packages {name: {version: {dep: req}}}, root deps, preferred
    ("newest", {"foo": {"1.0.0": {"bar": "^1"}, "1.1.0": {"bar": "^1"}},
                "bar": {"1.0.0": {}, "1.2.0": {}, "2.0.0": {}}},
     {"foo": "^1"}, {}),
    ("backtrack", {"foo": {"1.0.0": {"bar": "^1"}, "1.1.0": {"bar": "^2"}},
                   "bar": {"1.0.0": {}, "2.0.0": {}}},
     {"foo": "^1", "bar": "^1"}, {}),
    ("diamond", {"a": {"1.0.0": {"shared": ">=2, <4"}},
                 "b": {"1.0.0": {"shared": "^3"}},
                 "shared": {"2.0.0": {}, "3.0.0": {}, "3.5.0": {}, "4.0.0": {}}},
     {"a": "*", "b": "*"}, {}),
    ("prefer-locked", {"foo": {"1.0.0": {}, "1.1.0": {}, "1.2.0": {}}},
     {"foo": "^1"}, {"foo": "1.1.0"}),
    ("prefer-outranged", {"foo": {"1.0.0": {}, "1.1.0": {}, "1.2.0": {}}},
     {"foo": ">=1.2"}, {"foo": "1.1.0"}),
    ("prerelease-skipped", {"foo": {"1.0.0": {}, "2.0.0-rc.1": {}}}, {"foo": "^1"}, {}),
    ("prerelease-asked", {"foo": {"1.0.0": {}, "2.0.0-rc.1": {}}}, {"foo": ">=2.0.0-rc.1"}, {}),
    ("no-versions", {"foo": {"1.0.0": {"bar": "^2"}}, "bar": {"1.0.0": {}}}, {"foo": "^1"}, {}),
    ("two-requesters", {"foo": {"1.0.0": {"bar": "^1"}}, "baz": {"1.0.0": {"bar": "^2"}},
                        "bar": {"1.0.0": {}, "2.0.0": {}}},
     {"foo": "*", "baz": "*"}, {}),
    ("unknown-package", {"foo": {"1.0.0": {"nothere": "*"}}}, {"foo": "*"}, {}),
    ("deep-backtrack", {
        "a": {"1.0.0": {"b": "^1", "c": "^1"}, "2.0.0": {"b": "^2", "c": "^2"}},
        "b": {"1.0.0": {"d": "^1"}, "2.0.0": {"d": "^2"}},
        "c": {"1.0.0": {"d": "^1"}, "2.0.0": {"d": "^1"}},
        "d": {"1.0.0": {}, "2.0.0": {}}},
     {"a": "*"}, {}),
    ("many-versions", {"a": {f"1.{i}.0": {"c": "^1"} for i in range(50)},
                       "b": {"1.0.0": {"c": "^2"}},
                       "c": {"1.0.0": {}, "2.0.0": {}}},
     {"a": "*", "b": "*"}, {}),
    ("older-line-chosen", {"x": {"1.0.0": {"y": "~1.0"}, "1.5.0": {"y": "~1.1"}, "2.0.0": {"y": "^2"}},
                           "y": {"1.0.3": {}, "1.1.2": {"z": "^1"}, "2.0.0": {"z": "^2"}},
                           "z": {"1.0.0": {}}},
     {"x": "<2"}, {}),
]


# --- the oracle --------------------------------------------------------------

class PyGraph:
    def __init__(self, packages: Graph, prefer: dict[str, str]):
        self.p = {n: {Version.parse(v): {d: parse_requirement(r) for d, r in deps.items()}
                      for v, deps in vs.items()} for n, vs in packages.items()}
        self.prefer = {n: Version.parse(v) for n, v in prefer.items()}

    def versions(self, package):
        vs = sorted(self.p.get(package, {}), reverse=True)
        if package in self.prefer and self.prefer[package] in vs:
            vs.remove(self.prefer[package])
            vs.insert(0, self.prefer[package])
        return vs

    def dependencies(self, package, version):
        return self.p[package][version]


def oracle(name: str, packages: Graph, root_deps: dict[str, str], prefer: dict[str, str]) -> list[str]:
    g = PyGraph(packages, prefer)
    g.p["root"] = {Version(0, 0, 0): {d: parse_requirement(r) for d, r in root_deps.items()}}
    try:
        picked = Solver(g).solve("root", Version(0, 0, 0))
    except NoSolution as e:
        return [f"{name} fail"] + [f"{name} | {line}" for line in str(e).split("\n")]
    del picked["root"]
    return [f"{name} ok"] + [f"{name} {k} {v}" for k, v in sorted(picked.items())]


# --- the Metaxu program ------------------------------------------------------------

def _lit(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def program() -> str:
    lines = [
        "from std.semver import Version, parse_version, parse_requirement, version_to_string, version, range_any;",
        "from std.solve import Graph, Dep, graph_new, graph_add, dep, solve;",
        "from std.parse import split_on;",
        "",
        "fn v(s: string) -> Version {",
        "    match parse_version(s) { None => version(0, 0, 0), Some(x) => x }",
        "}",
        "fn d(name: string, req: string) -> Dep {",
        "    match parse_requirement(req) { None => dep(name, range_any()), Some(r) => dep(name, r) }",
        "}",
        "fn deps0() -> Vec { Vec.new() }",
        "fn deps1(a: Dep) -> Vec { let @mut x = Vec.new(); x.push(a); x }",
        "fn deps2(a: Dep, b: Dep) -> Vec { let @mut x = Vec.new(); x.push(a); x.push(b); x }",
        "fn sorted_names(names: Vec) -> Vec {",
        "    let @mut out = Vec.new();",
        "    let @mut i = 0;",
        "    while i < len(names) {",
        "        let n = names[i];",
        "        let @mut placed = Vec.new();",
        "        let @mut ins = false;",
        "        let @mut k = 0;",
        "        while k < len(out) {",
        "            if ins == false && less(n, out[k]) { placed.push(n); ins = true } else { () };",
        "            placed.push(out[k]);",
        "            k = k + 1",
        "        }",
        "        if ins == false { placed.push(n) } else { () };",
        "        out = placed;",
        "        i = i + 1",
        "    }",
        "    out",
        "}",
        "fn less(a: string, b: string) -> bool {",
        "    let table = \"abcdefghijklmnopqrstuvwxyz\";",
        "    let @mut i = 0;",
        "    let @mut result = len(a) < len(b);",
        "    let @mut decided = false;",
        "    let n = if len(a) < len(b) { len(a) } else { len(b) };",
        "    while i < n && decided == false {",
        "        let ra = rank(table, a[i]);",
        "        let rb = rank(table, b[i]);",
        "        if ra != rb { result = ra < rb; decided = true } else { () };",
        "        i = i + 1",
        "    }",
        "    result",
        "}",
        "fn rank(table: string, c: string) -> int {",
        "    let @mut i = 0;",
        "    let @mut r = 0 - 1;",
        "    while i < len(table) { if table[i] == c { r = i } else { () }; i = i + 1 }",
        "    r",
        "}",
        "fn report(label: string, g: Graph, prefer_names: Vec, prefer_versions: Vec) -> () {",
        "    let out = solve(g, \"root\", version(0, 0, 0), prefer_names, prefer_versions);",
        "    if out.ok {",
        "        print(label + \" ok\");",
        "        let names = sorted_names(out.names);",
        "        let @mut i = 0;",
        "        while i < len(names) {",
        "            let @mut j = 0;",
        "            while j < len(out.names) {",
        "                if out.names[j] == names[i] { print(label + \" \" + names[i] + \" \" + version_to_string(out.versions[j])) } else { () };",
        "                j = j + 1",
        "            };",
        "            i = i + 1",
        "        }",
        "    } else {",
        "        print(label + \" fail\");",
        "        let lines = split_on(out.explanation, \"\\n\");",
        "        let @mut i = 0;",
        "        while i < len(lines) { print(label + \" | \" + lines[i]); i = i + 1 }",
        "    }",
        "}",
        "fn main() -> int {",
    ]
    for name, packages, root_deps, prefer in SCENARIOS:
        gv = "g_" + name.replace("-", "_")
        lines.append(f"    let @mut {gv} = graph_new();")

        def deps_expr(deps: dict[str, str]) -> str:
            items = [f"d({_lit(dn)}, {_lit(dr)})" for dn, dr in deps.items()]
            if not items:
                return "deps0()"
            if len(items) == 1:
                return f"deps1({items[0]})"
            assert len(items) == 2, "extend deps helpers for wider graphs"
            return f"deps2({items[0]}, {items[1]})"

        for pkg, vs in packages.items():
            for ver, deps in vs.items():
                lines.append(f"    graph_add({gv}, {_lit(pkg)}, v({_lit(ver)}), {deps_expr(deps)});")
        lines.append(f"    graph_add({gv}, \"root\", version(0, 0, 0), {deps_expr(root_deps)});")
        pn = "pn_" + name.replace("-", "_")
        pv = "pv_" + name.replace("-", "_")
        lines.append(f"    let @mut {pn} = Vec.new();")
        lines.append(f"    let @mut {pv} = Vec.new();")
        for k, val in prefer.items():
            lines.append(f"    {pn}.push({_lit(k)});")
            lines.append(f"    {pv}.push(v({_lit(val)}));")
        lines.append(f"    report({_lit(name)}, {gv}, {pn}, {pv});")
    lines += ["    0", "}", ""]
    return "\n".join(lines)


@pytest.fixture(scope="module")
def transcript():
    result, out = interp_run(program())
    assert result == 0
    return out.rstrip("\n").split("\n")


def expected() -> list[str]:
    out: list[str] = []
    for name, packages, root_deps, prefer in SCENARIOS:
        out.extend(oracle(name, packages, root_deps, prefer))
    return out


@pytest.mark.parametrize("name", [s[0] for s in SCENARIOS])
def test_scenario_matches_the_python_solver(transcript, name):
    got = [l for l in transcript if l.split(" ", 1)[0] == name]
    want = [l for l in expected() if l.split(" ", 1)[0] == name]
    assert got == want


def test_nothing_unaccounted_for(transcript):
    assert len(transcript) == len(expected())


@needs_clang
def test_native_prints_what_the_interpreter_printed(tmp_path, transcript):
    # Native std.solve needed three backend fixes found by this module and
    # std.semver (docs/glade_in_metaxu.md): string indexing, nested enum
    # payload refinements, and kind specialization of helpers such as
    # `is_none(o: Option)` that meet two payload kinds.  This differential
    # is the pin.
    from metaxu.compiler.tests.test_codegen_llvm import llvm_from_source
    from metaxu.compiler.llvm_run import compile_and_run
    exit_code, stdout = compile_and_run(llvm_from_source(program()), "main",
                                        workdir=str(tmp_path), timeout=300)
    assert exit_code == 0
    assert stdout.rstrip("\n").split("\n") == transcript

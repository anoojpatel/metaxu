"""tap's version model and solver, with no git or registry involved.

The semver half pins the requirement dialect (chapter 15 documents it);
the PubGrub half runs the scenarios from the algorithm's own write-up
against an in-memory provider: no conflicts, a conflict that needs
backtracking, a diamond, prereleases, and an unsolvable graph whose
explanation must name the reason.
"""
from __future__ import annotations

import pytest

from metaxu.tap.pubgrub import NoSolution, Solver
from metaxu.tap.semver import Range, Version, VersionError, parse_requirement

V = Version.parse


# --- versions --------------------------------------------------------------

def test_version_ordering_follows_semver():
    order = ["0.9.9", "1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-beta", "1.0.0-rc.1",
             "1.0.0", "1.0.1", "1.1.0", "2.0.0"]
    parsed = [V(s) for s in order]
    assert parsed == sorted(parsed)
    assert str(V("1.2.3-rc.1+build.5")) == "1.2.3-rc.1"
    with pytest.raises(VersionError):
        V("1.2")


@pytest.mark.parametrize("req,inside,outside", [
    ("^1.2.3", ["1.2.3", "1.9.0"], ["1.2.2", "2.0.0"]),
    ("1.2.3", ["1.2.3", "1.9.0"], ["2.0.0"]),
    ("^0.2.3", ["0.2.3", "0.2.9"], ["0.3.0", "0.2.2"]),
    ("^0.0.3", ["0.0.3"], ["0.0.4", "0.0.2"]),
    ("^1", ["1.0.0", "1.99.0"], ["2.0.0", "0.9.0"]),
    ("~1.2.3", ["1.2.3", "1.2.9"], ["1.3.0"]),
    ("~1.2", ["1.2.0", "1.2.9"], ["1.3.0"]),
    ("~1", ["1.0.0", "1.9.9"], ["2.0.0"]),
    ("1.2.*", ["1.2.0", "1.2.9"], ["1.3.0", "1.1.9"]),
    ("1.*", ["1.0.0", "1.9.9"], ["2.0.0"]),
    ("*", ["0.0.1", "9.9.9"], []),
    ("=1.2.3", ["1.2.3"], ["1.2.4"]),
    (">=1.2, <2", ["1.2.0", "1.9.9"], ["1.1.9", "2.0.0"]),
    (">1.2", ["1.3.0"], ["1.2.9"]),
    ("<=1.2", ["1.2.9"], ["1.3.0"]),
    (">=1.0.0-rc.1", ["1.0.0-rc.1", "1.0.0-rc.2", "1.0.0", "1.1.0"], ["1.0.0-alpha", "0.9.0"]),
    ("^1.0", ["1.0.0"], ["2.0.0-rc.1", "1.5.0-beta"]),
])
def test_requirements(req, inside, outside):
    r = parse_requirement(req)
    for v in inside:
        assert r.contains(V(v)), f"{req} should contain {v}"
    for v in outside:
        assert not r.contains(V(v)), f"{req} should not contain {v}"


def test_range_algebra_is_closed():
    a = parse_requirement("^1")
    b = parse_requirement(">=1.5, <3")
    assert a.intersect(b) == parse_requirement(">=1.5, <2")
    assert a.union(b) == parse_requirement(">=1, <3")
    assert a.complement().contains(V("2.0.0")) and not a.complement().contains(V("1.5.0"))
    assert a.complement().complement() == a
    assert a.intersect(a.complement()).is_empty()
    assert a.union(a.complement()).is_any()
    assert Range.exact(V("1.2.3")).exact_version() == V("1.2.3")
    assert str(parse_requirement("^1.2")) == ">=1.2.0, <2.0.0"


# --- the solver ------------------------------------------------------------

class Graph:
    """An in-memory provider: {name: {version: {dep: requirement}}}."""

    def __init__(self, packages: dict[str, dict[str, dict[str, str]]], prefer: dict[str, str] | None = None):
        self.p = {n: {V(v): {d: parse_requirement(r) for d, r in deps.items()}
                      for v, deps in vs.items()} for n, vs in packages.items()}
        self.prefer = {n: V(v) for n, v in (prefer or {}).items()}
        self.asked: list[tuple[str, str]] = []

    def versions(self, package):
        vs = sorted(self.p.get(package, {}), reverse=True)
        if package in self.prefer and self.prefer[package] in vs:
            vs.remove(self.prefer[package])
            vs.insert(0, self.prefer[package])
        return vs

    def dependencies(self, package, version):
        self.asked.append((package, str(version)))
        return self.p[package][version]


LAST_SOLVER: list[Solver] = []


def solve(graph: Graph, root_deps: dict[str, str]) -> dict[str, str]:
    graph.p["root"] = {V("0.0.0"): {d: parse_requirement(r) for d, r in root_deps.items()}}
    solver = Solver(graph)
    LAST_SOLVER[:] = [solver]
    picked = solver.solve("root", V("0.0.0"))
    del picked["root"]
    return {k: str(v) for k, v in sorted(picked.items())}


def test_no_conflicts_picks_newest():
    g = Graph({"foo": {"1.0.0": {"bar": "^1"}, "1.1.0": {"bar": "^1"}},
               "bar": {"1.0.0": {}, "1.2.0": {}, "2.0.0": {}}})
    assert solve(g, {"foo": "^1"}) == {"bar": "1.2.0", "foo": "1.1.0"}


def test_backtracks_to_an_older_version_that_fits():
    # the newest foo wants bar 2, but root pins bar to 1: foo 1.0.0 it is
    g = Graph({"foo": {"1.0.0": {"bar": "^1"}, "1.1.0": {"bar": "^2"}},
               "bar": {"1.0.0": {}, "2.0.0": {}}})
    assert solve(g, {"foo": "^1", "bar": "^1"}) == {"bar": "1.0.0", "foo": "1.0.0"}


def test_diamond_agrees_on_one_shared_version():
    g = Graph({"a": {"1.0.0": {"shared": ">=2, <4"}},
               "b": {"1.0.0": {"shared": "^3"}},
               "shared": {"2.0.0": {}, "3.0.0": {}, "3.5.0": {}, "4.0.0": {}}})
    assert solve(g, {"a": "*", "b": "*"}) == {"a": "1.0.0", "b": "1.0.0", "shared": "3.5.0"}


def test_prefers_the_locked_version_when_it_still_fits():
    g = Graph({"foo": {"1.0.0": {}, "1.1.0": {}, "1.2.0": {}}}, prefer={"foo": "1.1.0"})
    assert solve(g, {"foo": "^1"}) == {"foo": "1.1.0"}
    g2 = Graph({"foo": {"1.0.0": {}, "1.1.0": {}, "1.2.0": {}}}, prefer={"foo": "1.1.0"})
    assert solve(g2, {"foo": ">=1.2"}) == {"foo": "1.2.0"}   # the lock cannot override the range


def test_prereleases_only_when_asked_for():
    g = Graph({"foo": {"1.0.0": {}, "2.0.0-rc.1": {}}})
    assert solve(g, {"foo": "^1"}) == {"foo": "1.0.0"}
    assert solve(g, {"foo": ">=2.0.0-rc.1"}) == {"foo": "2.0.0-rc.1"}


def test_unsolvable_graph_explains_itself():
    g = Graph({"foo": {"1.0.0": {"bar": "^2"}},
               "bar": {"1.0.0": {}}})
    with pytest.raises(NoSolution) as exc:
        solve(g, {"foo": "^1"})
    text = str(exc.value)
    assert "foo 1.0.0 depends on bar >=2.0.0, <3.0.0" in text
    assert "no versions of bar match" in text


def test_conflicting_requirements_are_explained_in_terms_of_the_root():
    g = Graph({"foo": {"1.0.0": {"bar": "^1"}},
               "baz": {"1.0.0": {"bar": "^2"}},
               "bar": {"1.0.0": {}, "2.0.0": {}}})
    with pytest.raises(NoSolution) as exc:
        solve(g, {"foo": "*", "baz": "*"})
    text = str(exc.value)
    assert "foo 1.0.0 depends on bar" in text and "baz 1.0.0 depends on bar" in text


def test_pubgrub_does_not_enumerate_dead_ends():
    """Many versions of `a` all pin `c` to 1; `b` needs `c` 2. A
    backtracking resolver tries every `a`; PubGrub learns once that any
    `a` conflicts and stops."""
    versions = {f"1.{i}.0": {"c": "^1"} for i in range(50)}
    g = Graph({"a": versions, "b": {"1.0.0": {"c": "^2"}},
               "c": {"1.0.0": {}, "2.0.0": {}}})
    with pytest.raises(NoSolution) as exc:
        solve(g, {"a": "*", "b": "*"})
    # one widened fact covers every `a`; no per-version conflict rounds
    assert len(LAST_SOLVER[0].incompats) < 12
    assert "a >=1.0.0, <=1.49.0 depends on c >=1.0.0, <2.0.0" in str(exc.value)

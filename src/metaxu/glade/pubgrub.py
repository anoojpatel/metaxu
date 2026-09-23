"""PubGrub version solving.

This is the algorithm behind Dart's pub, Cargo's resolver and uv,
written from the published description
(https://github.com/dart-lang/pub/blob/master/doc/solver.md).  It is a
CDCL SAT solver specialised to package versions: it propagates the
consequences of every choice, and when it hits a contradiction it
learns a new fact (an *incompatibility*) that explains the conflict
and jumps back to where that fact first mattered, instead of trying
versions one by one.  Two properties follow that a backtracking
resolver does not have: it rarely revisits a dead end, and when
resolution is impossible it can say exactly why, as a chain of
"because A depends on B and B depends on C ..." sentences.

The solver knows nothing about registries or git.  It asks a
`Provider` two questions: which versions of a package exist (newest
preferred first) and what a given version depends on.

    solver = Solver(provider)
    picked = solver.solve("root", Version(0, 0, 0))   # {name: Version}

`solve` raises `NoSolution`, whose message is the explanation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .semver import Range, Version


class Provider(Protocol):
    def versions(self, package: str) -> list[Version]:
        """All known versions, best candidate first (a locked or newest
        version), excluding nothing: the solver filters by range."""

    def dependencies(self, package: str, version: Version) -> dict[str, Range] | None:
        """name -> allowed range, or None when the version cannot be
        used (its manifest is missing or broken)."""


# --- terms and incompatibilities ---------------------------------------------

@dataclass(frozen=True)
class Term:
    """`package` is (positive) or is not (negative) in `range`."""
    package: str
    range: Range
    positive: bool = True

    @property
    def allowed(self) -> Range:
        """The set of versions this term permits."""
        return self.range if self.positive else self.range.complement()

    def negate(self) -> "Term":
        return Term(self.package, self.range, not self.positive)

    def intersect(self, other: "Term") -> "Term":
        assert self.package == other.package
        if self.positive and other.positive:
            return Term(self.package, self.range.intersect(other.range), True)
        if not self.positive and not other.positive:
            return Term(self.package, self.range.union(other.range), False)
        pos, neg = (self, other) if self.positive else (other, self)
        return Term(self.package, pos.range.intersect(neg.range.complement()), True)

    def satisfies(self, other: "Term") -> bool:
        """Every version this term allows, `other` allows too."""
        return self.allowed.is_subset_of(other.allowed)

    def contradicts(self, other: "Term") -> bool:
        return self.allowed.is_disjoint_from(other.allowed)

    def __str__(self) -> str:
        if self.positive:
            return f"{self.package} {self.range}"
        return f"not {self.package} {self.range}"


@dataclass(frozen=True)
class Cause:
    kind: str                      # root | dependency | no_versions | unavailable | derived
    package: str = ""
    version: Version | None = None
    dependency: str = ""
    left: "Incompatibility | None" = None
    right: "Incompatibility | None" = None


@dataclass(frozen=True)
class Incompatibility:
    """A set of terms that cannot all be true at once."""
    terms: tuple[Term, ...]
    cause: Cause

    def term_for(self, package: str) -> Term | None:
        for t in self.terms:
            if t.package == package:
                return t
        return None

    @property
    def packages(self) -> list[str]:
        return [t.package for t in self.terms]

    def describe(self) -> str:
        c = self.cause
        if c.kind == "dependency":
            dep = self.term_for(c.dependency)
            own = self.term_for(c.package)
            assert dep is not None and own is not None
            who = str(c.version) if own.range.exact_version() is not None else str(own.range)
            return f"{c.package} {who} depends on {c.dependency} {dep.range}"
        if c.kind == "no_versions":
            t = self.terms[0]
            return f"no versions of {t.package} match {t.range}"
        if c.kind == "unavailable":
            return f"{c.package} {c.version} cannot be used (its manifest is missing or broken)"
        if c.kind == "root":
            return f"{c.package} is the project"
        # derived
        if not self.terms:
            return "version solving failed"
        if len(self.terms) == 1:
            t = self.terms[0]
            if t.positive:
                return f"{t.package} {t.range} is forbidden"
            return f"{t.package} must be {t.range}"
        pos = [t for t in self.terms if t.positive]
        neg = [t for t in self.terms if not t.positive]
        if len(pos) == 1 and len(neg) == 1:
            return f"{pos[0].package} {pos[0].range} requires {neg[0].package} {neg[0].range}"
        return " and ".join(str(t) for t in self.terms) + " are incompatible"


# --- the partial solution -----------------------------------------------------

@dataclass
class Assignment:
    term: Term
    decision_level: int
    index: int
    cause: Incompatibility | None       # None for a decision

    @property
    def is_decision(self) -> bool:
        return self.cause is None


@dataclass
class PartialSolution:
    assignments: list[Assignment] = field(default_factory=list)
    decisions: dict[str, Version] = field(default_factory=dict)
    derivations: dict[str, Term] = field(default_factory=dict)   # package -> intersection
    decision_level: int = 0

    def _add(self, term: Term, cause: Incompatibility | None) -> None:
        self.assignments.append(Assignment(term, self.decision_level, len(self.assignments), cause))
        prior = self.derivations.get(term.package)
        self.derivations[term.package] = term if prior is None else prior.intersect(term)

    def decide(self, package: str, version: Version) -> None:
        self.decision_level += 1
        self.decisions[package] = version
        self._add(Term(package, Range.exact(version), True), None)

    def derive(self, term: Term, cause: Incompatibility) -> None:
        self._add(term, cause)

    def relation(self, term: Term) -> str:
        """subset (the solution implies term), disjoint (contradicts), overlapping."""
        have = self.derivations.get(term.package)
        if have is None:
            return "overlapping"
        if have.satisfies(term):
            return "subset"
        if have.contradicts(term):
            return "disjoint"
        return "overlapping"

    def relation_of(self, incompat: Incompatibility) -> tuple[str, Term | None]:
        """satisfied, contradicted, almost (with the one unsatisfied term),
        or inconclusive."""
        unsatisfied: Term | None = None
        for t in incompat.terms:
            rel = self.relation(t)
            if rel == "disjoint":
                return "contradicted", t
            if rel == "overlapping":
                if unsatisfied is not None:
                    return "inconclusive", None
                unsatisfied = t
        return ("satisfied", None) if unsatisfied is None else ("almost", unsatisfied)

    def undecided_positive(self) -> list[str]:
        return [p for p, t in self.derivations.items()
                if t.positive and p not in self.decisions]

    def backtrack(self, level: int) -> None:
        keep = [a for a in self.assignments if a.decision_level <= level]
        self.assignments = []
        self.decisions = {}
        self.derivations = {}
        self.decision_level = level
        for a in keep:
            if a.is_decision:
                self.decisions[a.term.package] = a.term.range.exact_version()  # type: ignore[assignment]
            self.assignments.append(Assignment(a.term, a.decision_level, len(self.assignments), a.cause))
            prior = self.derivations.get(a.term.package)
            self.derivations[a.term.package] = a.term if prior is None else prior.intersect(a.term)

    def satisfier(self, incompat: Incompatibility, extra: Assignment | None = None) -> Assignment:
        """The earliest assignment such that the assignments up to and
        including it (plus `extra`, if given) satisfy the incompatibility."""
        result: Assignment | None = None
        for term in incompat.terms:
            acc: Term | None = None
            if extra is not None and extra.term.package == term.package:
                acc = extra.term
                if acc.satisfies(term):
                    continue
            found = None
            for a in self.assignments:
                if a.term.package != term.package:
                    continue
                if extra is not None and a.index >= extra.index:
                    break
                acc = a.term if acc is None else acc.intersect(a.term)
                if acc.satisfies(term):
                    found = a
                    break
            if found is None:
                raise AssertionError(f"incompatibility not satisfied: {incompat.describe()}")
            if result is None or found.index > result.index:
                result = found
        assert result is not None
        return result


# --- the solver -----------------------------------------------------------------

class NoSolution(Exception):
    def __init__(self, incompat: Incompatibility):
        self.incompat = incompat
        super().__init__(Reporter().explain(incompat))


class Solver:
    def __init__(self, provider: Provider):
        self.provider = provider
        self.solution = PartialSolution()
        self.incompats: list[Incompatibility] = []
        self.by_package: dict[str, list[Incompatibility]] = {}
        self._added: set[tuple[str, Version]] = set()
        self._deps: dict[tuple[str, Version], dict[str, Range] | None] = {}

    def _dependencies(self, package: str, version: Version) -> dict[str, Range] | None:
        key = (package, version)
        if key not in self._deps:
            self._deps[key] = self.provider.dependencies(package, version)
        return self._deps[key]

    def _widen(self, package: str, version: Version, dep: str, rng: Range) -> Range:
        """The contiguous run of versions around `version` that all
        depend on `dep` with exactly this range.

        One incompatibility "a >=1.0.0, <=1.49.0 depends on c ^1" does
        the work of fifty per-version ones: a conflict with c is learned
        once for the whole run instead of being rediscovered version by
        version. This is the step that keeps PubGrub from enumerating
        dead ends, and it is cheap because a registry index lists every
        version's dependencies up front."""
        ordered = sorted(self.provider.versions(package))
        i = ordered.index(version)
        lo = hi = i
        while lo > 0 and (self._dependencies(package, ordered[lo - 1]) or {}).get(dep) == rng:
            lo -= 1
        while hi + 1 < len(ordered) and (self._dependencies(package, ordered[hi + 1]) or {}).get(dep) == rng:
            hi += 1
        if lo == hi:
            return Range.exact(version)
        return Range.between(ordered[lo], ordered[hi], lo_inc=True, hi_inc=True)

    def _add_incompat(self, inc: Incompatibility) -> None:
        self.incompats.append(inc)
        for p in inc.packages:
            self.by_package.setdefault(p, []).append(inc)

    def solve(self, root: str, root_version: Version) -> dict[str, Version]:
        self._add_incompat(Incompatibility(
            (Term(root, Range.exact(root_version), False),), Cause("root", package=root)))
        nxt: str | None = root
        while nxt is not None:
            self._propagate(nxt)
            nxt = self._choose()
        return dict(self.solution.decisions)

    # unit propagation
    def _propagate(self, package: str) -> None:
        changed = [package]
        while changed:
            pkg = changed.pop()
            for inc in reversed(self.by_package.get(pkg, [])):
                rel, term = self.solution.relation_of(inc)
                if rel == "satisfied":
                    root_cause = self._resolve_conflict(inc)
                    rel2, term2 = self.solution.relation_of(root_cause)
                    assert rel2 == "almost" and term2 is not None, rel2
                    self.solution.derive(term2.negate(), root_cause)
                    changed = [term2.package]
                    break
                if rel == "almost" and term is not None:
                    self.solution.derive(term.negate(), inc)
                    changed.append(term.package)

    # conflict resolution
    def _resolve_conflict(self, inc: Incompatibility) -> Incompatibility:
        new = False
        while True:
            if not inc.terms or (len(inc.terms) == 1 and inc.terms[0].positive
                                 and inc.terms[0].package == self.incompats[0].terms[0].package):
                raise NoSolution(inc)
            satisfier = self.solution.satisfier(inc)
            term = inc.term_for(satisfier.term.package)
            assert term is not None
            # the previous satisfier: the latest assignment (other than the
            # satisfier) still needed for the incompatibility to hold
            prev_level = 1
            if len(inc.terms) > 1 or not satisfier.term.satisfies(term):
                try:
                    prev = self._previous_satisfier(inc, satisfier)
                    prev_level = prev.decision_level if prev is not None else 1
                except AssertionError:
                    prev_level = 1
            if satisfier.is_decision or prev_level != satisfier.decision_level:
                if new:
                    self._add_incompat(inc)
                self.solution.backtrack(prev_level)
                return inc
            # derive a new incompatibility from this one and the satisfier's cause
            assert satisfier.cause is not None
            terms: dict[str, Term] = {}
            for t in (*inc.terms, *satisfier.cause.terms):
                if t.package == satisfier.term.package:
                    continue
                terms[t.package] = t if t.package not in terms else terms[t.package].intersect(t)
            if not satisfier.term.satisfies(term):
                leftover = satisfier.term.intersect(term.negate())
                terms[satisfier.term.package] = leftover.negate()
            inc = Incompatibility(tuple(terms.values()),
                                  Cause("derived", left=inc, right=satisfier.cause))
            new = True

    def _previous_satisfier(self, inc: Incompatibility, satisfier: Assignment) -> Assignment | None:
        """Earliest assignment before `satisfier` such that it plus the
        satisfier satisfy `inc`; None when the satisfier alone does."""
        best: Assignment | None = None
        for term in inc.terms:
            acc: Term | None = None
            if term.package == satisfier.term.package:
                acc = satisfier.term
                if acc.satisfies(term):
                    continue
            found = None
            for a in self.solution.assignments:
                if a.index >= satisfier.index or a.term.package != term.package:
                    continue
                acc = a.term if acc is None else acc.intersect(a.term)
                if acc.satisfies(term):
                    found = a
                    break
            assert found is not None
            if best is None or found.index > best.index:
                best = found
        return best

    # decision making
    def _choose(self) -> str | None:
        candidates = self.solution.undecided_positive()
        if not candidates:
            return None
        best_pkg, best_versions, best_term = None, None, None
        for pkg in candidates:
            term = self.solution.derivations[pkg]
            versions = [v for v in self.provider.versions(pkg) if term.range.contains(v)]
            if best_versions is None or len(versions) < len(best_versions):
                best_pkg, best_versions, best_term = pkg, versions, term
        assert best_pkg is not None and best_versions is not None and best_term is not None
        if not best_versions:
            self._add_incompat(Incompatibility((Term(best_pkg, best_term.range, True),),
                                               Cause("no_versions")))
            return best_pkg
        version = best_versions[0]
        deps = self._dependencies(best_pkg, version)
        if deps is None:
            self._add_incompat(Incompatibility((Term(best_pkg, Range.exact(version), True),),
                                               Cause("unavailable", package=best_pkg, version=version)))
            return best_pkg
        conflict = False
        if (best_pkg, version) not in self._added:
            self._added.add((best_pkg, version))
            for dep, rng in deps.items():
                covered = self._widen(best_pkg, version, dep, rng)
                inc = Incompatibility(
                    (Term(best_pkg, covered, True), Term(dep, rng, False)),
                    Cause("dependency", package=best_pkg, version=version, dependency=dep))
                self._add_incompat(inc)
                # would this incompatibility be satisfied by the decision?
                rel = self.solution.relation(Term(dep, rng, False))
                if rel == "subset":
                    conflict = True
        if not conflict:
            self.solution.decide(best_pkg, version)
        return best_pkg


# --- explanations -----------------------------------------------------------------

class Reporter:
    """Turn a derivation tree into sentences a person can act on."""

    def explain(self, inc: Incompatibility) -> str:
        lines: list[str] = []
        self._write(inc, lines)
        return "\n".join(lines)

    def _write(self, inc: Incompatibility, lines: list[str]) -> None:
        c = inc.cause
        if c.kind != "derived":
            lines.append(inc.describe() + ".")
            return
        assert c.left is not None and c.right is not None
        l_der, r_der = c.left.cause.kind == "derived", c.right.cause.kind == "derived"
        if l_der and r_der:
            self._write(c.left, lines)
            self._write(c.right, lines)
            lines.append(f"Thus, {inc.describe()}.")
        elif l_der or r_der:
            derived, ext = (c.left, c.right) if l_der else (c.right, c.left)
            self._write(derived, lines)
            lines.append(f"And because {ext.describe()}, {inc.describe()}.")
        else:
            lines.append(f"Because {c.left.describe()} and {c.right.describe()}, {inc.describe()}.")

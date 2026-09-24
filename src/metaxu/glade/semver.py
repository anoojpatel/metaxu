"""Semantic versions and version requirements.

A `Version` is MAJOR.MINOR.PATCH with an optional prerelease
(`1.2.0-beta.1`); ordering follows semver.org, so prereleases sort
before the release they precede.

A `Range` is a set of versions: a sorted list of disjoint intervals
with open or closed ends, closed under intersection, union and
complement.  That closure is what the solver needs (a "not this range"
term is a complement), and it is why requirements are parsed into
ranges instead of being matched one comparator at a time.

Requirement syntax, the Cargo dialect:

    ^1.2.3   >=1.2.3, <2.0.0    (^0.2.3 is <0.3.0; ^0.0.3 is <0.0.4)
    1.2.3    same as ^1.2.3     (a bare version means "compatible with")
    ~1.2.3   >=1.2.3, <1.3.0    (~1.2 is <1.3.0; ~1 is <2.0.0)
    1.2.*    >=1.2.0, <1.3.0    (1.* and * likewise)
    =1.2.3   exactly 1.2.3
    >=1.2, <2   comparators, comma means AND

Prerelease versions are excluded from every range unless a bound of
the range is itself a prerelease of the same MAJOR.MINOR.PATCH, so
`^1.0` never picks `2.0.0-rc.1` and `>=1.0.0-rc.1` does pick `1.0.0-rc.2`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import total_ordering
from typing import Iterable

_VERSION_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z.-]+))?(?:\+([0-9A-Za-z.-]+))?$")
_PARTIAL_RE = re.compile(
    r"^(\d+)(?:\.(\d+|\*))?(?:\.(\d+|\*))?(?:-([0-9A-Za-z.-]+))?$")


class VersionError(ValueError):
    pass


@total_ordering
@dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int
    pre: tuple[int | str, ...] = ()

    @classmethod
    def parse(cls, text: str) -> "Version":
        m = _VERSION_RE.match(text.strip())
        if not m:
            raise VersionError(f"{text!r} is not a version (MAJOR.MINOR.PATCH[-pre])")
        pre = tuple(int(p) if p.isdigit() else p for p in m.group(4).split(".")) if m.group(4) else ()
        # dotted identifiers are non-empty (semver.org item 9 and 10)
        if any(p == "" for p in pre) or (m.group(5) and "" in m.group(5).split(".")):
            raise VersionError(f"{text!r} has an empty identifier")
        return cls(int(m.group(1)), int(m.group(2)), int(m.group(3)), pre)

    @property
    def triple(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def _key(self):
        # a release sorts after every prerelease of the same triple;
        # numeric identifiers sort before alphanumeric ones
        pre_key = tuple((0, p, "") if isinstance(p, int) else (1, 0, p) for p in self.pre)
        return (self.triple, 1 if not self.pre else 0, pre_key)

    def __lt__(self, other: "Version") -> bool:
        return self._key() < other._key()

    def __str__(self) -> str:
        s = f"{self.major}.{self.minor}.{self.patch}"
        return s + ("-" + ".".join(str(p) for p in self.pre) if self.pre else "")

    def bump_patch(self) -> "Version":
        return Version(self.major, self.minor, self.patch + 1)


# --- intervals --------------------------------------------------------------

@dataclass(frozen=True)
class Interval:
    """lo <= v <= hi with each end open or closed; None means unbounded."""
    lo: Version | None
    lo_inc: bool
    hi: Version | None
    hi_inc: bool

    def contains(self, v: Version) -> bool:
        if self.lo is not None and (v < self.lo or (v == self.lo and not self.lo_inc)):
            return False
        if self.hi is not None and (v > self.hi or (v == self.hi and not self.hi_inc)):
            return False
        return True

    def is_empty(self) -> bool:
        if self.lo is None or self.hi is None:
            return False
        if self.lo < self.hi:
            return False
        return not (self.lo == self.hi and self.lo_inc and self.hi_inc)

    def __str__(self) -> str:
        if self.lo is not None and self.lo == self.hi:
            return f"={self.lo}"
        parts = []
        if self.lo is not None:
            parts.append((">=" if self.lo_inc else ">") + str(self.lo))
        if self.hi is not None:
            parts.append(("<=" if self.hi_inc else "<") + str(self.hi))
        return ", ".join(parts) if parts else "*"


def _lo_after(a: Interval, b: Interval) -> bool:
    """a's low end is at or after b's low end."""
    if b.lo is None:
        return True
    if a.lo is None:
        return False
    return a.lo > b.lo or (a.lo == b.lo and (not a.lo_inc or b.lo_inc))


def _hi_before(a: Interval, b: Interval) -> bool:
    if b.hi is None:
        return True
    if a.hi is None:
        return False
    return a.hi < b.hi or (a.hi == b.hi and (not a.hi_inc or b.hi_inc))


def _intersect(a: Interval, b: Interval) -> Interval | None:
    lo = a if _lo_after(a, b) else b
    hi = a if _hi_before(a, b) else b
    r = Interval(lo.lo, lo.lo_inc, hi.hi, hi.hi_inc)
    return None if r.is_empty() else r


def _touch_or_overlap(a: Interval, b: Interval) -> bool:
    """a and b (a starting no later than b) form one contiguous interval."""
    if a.hi is None or b.lo is None:
        return True
    if a.hi > b.lo:
        return True
    return a.hi == b.lo and (a.hi_inc or b.lo_inc)


class Range:
    """A set of versions as disjoint, sorted intervals."""

    __slots__ = ("intervals",)

    def __init__(self, intervals: Iterable[Interval] = ()):
        merged: list[Interval] = []
        for iv in sorted((i for i in intervals if not i.is_empty()),
                         key=lambda i: (i.lo is not None, i.lo or Version(0, 0, 0), not i.lo_inc)):
            if merged and _touch_or_overlap(merged[-1], iv):
                last = merged[-1]
                hi = last if _hi_before(iv, last) else iv
                merged[-1] = Interval(last.lo, last.lo_inc, hi.hi, hi.hi_inc)
            else:
                merged.append(iv)
        self.intervals: tuple[Interval, ...] = tuple(merged)

    # constructors
    @classmethod
    def any(cls) -> "Range":
        return cls([Interval(None, True, None, True)])

    @classmethod
    def empty(cls) -> "Range":
        return cls()

    @classmethod
    def exact(cls, v: Version) -> "Range":
        return cls([Interval(v, True, v, True)])

    @classmethod
    def between(cls, lo: Version | None, hi: Version | None,
                lo_inc: bool = True, hi_inc: bool = False) -> "Range":
        return cls([Interval(lo, lo_inc, hi, hi_inc)])

    # set algebra
    def intersect(self, other: "Range") -> "Range":
        out = []
        for a in self.intervals:
            for b in other.intervals:
                r = _intersect(a, b)
                if r is not None:
                    out.append(r)
        return Range(out)

    def union(self, other: "Range") -> "Range":
        return Range((*self.intervals, *other.intervals))

    def complement(self) -> "Range":
        if not self.intervals:
            return Range.any()
        out = []
        prev_hi: Version | None = None
        prev_hi_inc = False
        first = True
        for iv in self.intervals:
            if first:
                if iv.lo is not None:
                    out.append(Interval(None, True, iv.lo, not iv.lo_inc))
                first = False
            else:
                out.append(Interval(prev_hi, not prev_hi_inc, iv.lo, not iv.lo_inc))
            prev_hi, prev_hi_inc = iv.hi, iv.hi_inc
        if prev_hi is not None:
            out.append(Interval(prev_hi, not prev_hi_inc, None, True))
        return Range(out)

    def is_empty(self) -> bool:
        return not self.intervals

    def is_any(self) -> bool:
        return len(self.intervals) == 1 and self.intervals[0] == Interval(None, True, None, True)

    def is_subset_of(self, other: "Range") -> bool:
        return self.intersect(other.complement()).is_empty()

    def is_disjoint_from(self, other: "Range") -> bool:
        return self.intersect(other).is_empty()

    def contains(self, v: Version) -> bool:
        for iv in self.intervals:
            if not iv.contains(v):
                continue
            if not v.pre:
                return True
            # a prerelease only matches a range that names a prerelease
            # of the same triple at one of its ends
            for bound in (iv.lo, iv.hi):
                if bound is not None and bound.pre and bound.triple == v.triple:
                    return True
        return False

    def exact_version(self) -> Version | None:
        if len(self.intervals) == 1:
            iv = self.intervals[0]
            if iv.lo is not None and iv.lo == iv.hi and iv.lo_inc and iv.hi_inc:
                return iv.lo
        return None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Range) and self.intervals == other.intervals

    def __hash__(self) -> int:
        return hash(self.intervals)

    def __str__(self) -> str:
        if self.is_any():
            return "*"
        if not self.intervals:
            return "(none)"
        return " or ".join(str(iv) for iv in self.intervals)

    __repr__ = __str__


# --- requirement parsing ----------------------------------------------------

def _partial(text: str) -> tuple[int, int | None, int | None, tuple]:
    m = _PARTIAL_RE.match(text)
    if not m:
        raise VersionError(f"{text!r} is not a version requirement")
    major = int(m.group(1))
    minor = None if m.group(2) in (None, "*") else int(m.group(2))
    patch = None if m.group(3) in (None, "*") else int(m.group(3))
    if minor is None and patch is not None:
        raise VersionError(f"{text!r}: a patch needs a minor")
    pre = tuple(int(p) if p.isdigit() else p for p in m.group(4).split(".")) if m.group(4) else ()
    if any(p == "" for p in pre):
        raise VersionError(f"{text!r} has an empty prerelease identifier")
    return major, minor, patch, pre


def _caret(text: str) -> Range:
    major, minor, patch, pre = _partial(text)
    lo = Version(major, minor or 0, patch or 0, pre)
    if major > 0 or minor is None:
        hi = Version(major + 1, 0, 0)
    elif minor > 0 or patch is None:
        hi = Version(0, minor + 1, 0)
    else:
        hi = Version(0, 0, patch + 1)
    return Range.between(lo, hi)


def _tilde(text: str) -> Range:
    major, minor, patch, pre = _partial(text)
    lo = Version(major, minor or 0, patch or 0, pre)
    hi = Version(major + 1, 0, 0) if minor is None else Version(major, minor + 1, 0)
    return Range.between(lo, hi)


def _wild_or_exact(text: str) -> Range:
    major, minor, patch, pre = _partial(text)
    if patch is not None:
        return Range.exact(Version(major, minor, patch, pre))
    lo = Version(major, minor or 0, 0)
    hi = Version(major + 1, 0, 0) if minor is None else Version(major, minor + 1, 0)
    return Range.between(lo, hi)


def _comparator(op: str, text: str) -> Range:
    major, minor, patch, pre = _partial(text)
    if patch is not None:
        v = Version(major, minor, patch, pre)
        return {">=": Range.between(v, None),
                ">": Range.between(v, None, lo_inc=False),
                "<": Range.between(None, v),
                "<=": Range.between(None, v, hi_inc=True),
                "=": Range.exact(v)}[op]
    # a partial version is the whole span it names
    span = _wild_or_exact(text)
    lo, hi = span.intervals[0].lo, span.intervals[0].hi
    assert lo is not None and hi is not None
    return {">=": Range.between(lo, None),
            ">": Range.between(hi, None),
            "<": Range.between(None, lo),
            "<=": Range.between(None, hi),
            "=": span}[op]


def parse_requirement(text: str) -> Range:
    text = text.strip()
    if text in ("", "*"):
        return Range.any()
    result = Range.any()
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise VersionError(f"{text!r}: empty comparator")
        if part.startswith("^"):
            r = _caret(part[1:].strip())
        elif part.startswith("~"):
            r = _tilde(part[1:].strip())
        elif part.startswith((">=", "<=")):
            r = _comparator(part[:2], part[2:].strip())
        elif part.startswith((">", "<", "=")):
            r = _comparator(part[0], part[1:].strip())
        elif "*" in part:
            r = Range.any() if part == "*" else _wild_or_exact(part)
        else:
            r = _caret(part)
        result = result.intersect(r)
    return result

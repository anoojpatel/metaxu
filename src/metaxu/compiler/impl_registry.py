from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from .constraints import InstanceHead, ClassConstraint, FDMap


@dataclass(slots=True)
class TraitInfo:
    name: str
    fd_det: tuple[int, ...]
    fd_detd: tuple[int, ...]
    dict_type_name: str  # e.g., TDict


class ImplRegistry:
    """In-memory registry for trait dicts and instances.

    The only allowed global mutable state in the compiler pipeline.
    """

    def __init__(self) -> None:
        self.traits: Dict[str, TraitInfo] = {}
        self.instances: list[InstanceHead] = []

    def fd_map(self) -> FDMap:
        fds = {t.name: (t.fd_det, t.fd_detd) for t in self.traits.values()}
        return FDMap(fds)

    def register_trait(self, name: str, det: Sequence[int], detd: Sequence[int], dict_type_name: str) -> None:
        self.traits[name] = TraitInfo(name=name, fd_det=tuple(det), fd_detd=tuple(detd), dict_type_name=dict_type_name)

    def register_instance(self, inst: InstanceHead) -> None:
        # Coherence checks (consistency/coverage) would go here.
        self.instances.append(inst)

    def lookup_instances(self, name: str) -> list[InstanceHead]:
        return [i for i in self.instances if i.name == name]


# A single shared registry for now.
GLOBAL_IMPL_REGISTRY = ImplRegistry()

"""RetrievalScope: navigation scope over namespaces (front #3).

The scope an agent carries is NAVIGATION ONLY — which namespace subtrees
it sees inside a vault. The vault itself is never part of the scope: it
is a mount, bound at store construction, so the hard isolation boundary
cannot leak through a query-time object (the pattern every reference
system uses — Obsidian's ``path:`` operator never names a vault, the
memory tool's model never chooses its root).

Semantics, fixed in docs/plans/2026-08-28-v020-3-vault-namespace.md:

- namespaces are materialized paths (``/campanha-1/sessoes``);
- empty ``include`` means the whole vault;
- ``exclude`` ALWAYS wins over ``include``;
- prefix matching is boundary-aware (``/campanha-1`` matches itself and
  descendants, never ``/campanha-10``);
- :meth:`intersect` only narrows — a child scope can never widen the
  parent's; disjoint non-empty includes intersect to a scope that
  matches nothing.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

DEFAULT_VAULT = "__default__"
"""Vault stamped on data created before vaults existed (Pinecone's dunder
convention — cannot collide with a legitimate user vault named "default")."""

ROOT_NAMESPACE = "/"


def validate_vault(vault: str) -> str:
    """A vault is a flat mount name.

    Non-empty, no ``/`` (that is namespace syntax) and no ``:`` (the key
    separator on Redis, where the vault is part of the key path). Checked
    once, at the boundary: on ``ContextItem`` and on every store mount —
    a bad name must fail at construction, never on the read after the
    write.
    """
    if not vault.strip():
        msg = "vault must not be empty"
        raise ValueError(msg)
    if "/" in vault:
        msg = "vault is a flat mount name — '/' belongs to namespaces"
        raise ValueError(msg)
    if ":" in vault:
        msg = "vault must not contain ':' (it is a key separator on Redis)"
        raise ValueError(msg)
    return vault


def normalize_namespace(namespace: str) -> str:
    """Canonical form: leading ``/``, no trailing ``/`` (root stays ``/``).

    Raises ``ValueError`` on empty input — an empty namespace is always a
    caller bug, never a meaning.
    """
    ns = namespace.strip()
    if not ns:
        msg = "namespace must not be empty (root is '/')"
        raise ValueError(msg)
    if not ns.startswith("/"):
        ns = "/" + ns
    while len(ns) > 1 and ns.endswith("/"):
        ns = ns[:-1]
    return ns


def is_under(namespace: str, prefix: str) -> bool:
    """Boundary-aware descendant test over canonical paths.

    ``/campanha-1`` is under ``/campanha-1`` and ``/``; ``/campanha-10``
    is NOT under ``/campanha-1``.
    """
    if prefix == ROOT_NAMESPACE:
        return True
    return namespace == prefix or namespace.startswith(prefix + "/")


class RetrievalScope(BaseModel, frozen=True):
    """Which namespace subtrees are visible. Exclude wins; empty = all."""

    model_config = ConfigDict(frozen=True)

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    @field_validator("include", "exclude")
    @classmethod
    def _normalize(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_namespace(p) for p in paths)

    def matches(self, namespace: str) -> bool:
        """True when *namespace* is visible under this scope."""
        ns = normalize_namespace(namespace)
        if any(is_under(ns, p) for p in self.exclude):
            return False
        if not self.include:
            return True
        return any(is_under(ns, p) for p in self.include)

    def intersect(self, child: RetrievalScope) -> RetrievalScope:
        """Effective scope of a child under this parent — only narrows.

        - excludes are the union of both (an exclusion never comes back);
        - includes keep, for every parent x child pair, the DEEPER prefix
          when one contains the other; an empty side means "everything",
          so the other side's includes stand alone;
        - two non-empty, fully disjoint include sets have no visible
          namespace at all — expressed as ``exclude=("/",)`` so
          :meth:`matches` naturally refuses everything.
        """
        exclude = tuple(dict.fromkeys(self.exclude + child.exclude))
        if not self.include:
            include = child.include
        elif not child.include:
            include = self.include
        else:
            narrowed: list[str] = []
            for p in self.include:
                for c in child.include:
                    if is_under(c, p):
                        narrowed.append(c)
                    elif is_under(p, c):
                        narrowed.append(p)
            if not narrowed:
                return RetrievalScope(exclude=(ROOT_NAMESPACE,))
            include = tuple(dict.fromkeys(narrowed))
        return RetrievalScope(include=include, exclude=exclude)


# The scope published by the running agent turn — for the duration of a
# tool call and of the turn's own pipeline build (Agent.with_scope, and a
# parent's scope seen by a subagent). Read through active_scope() /
# effective_scope(); a child's effective scope is published ∩ own, so it
# can only narrow.
ACTIVE_SCOPE: ContextVar[RetrievalScope | None] = ContextVar(
    "anchor_active_scope", default=None,
)


def active_scope() -> RetrievalScope | None:
    """The scope published for the current agent turn, ``None`` outside one."""
    return ACTIVE_SCOPE.get()


def effective_scope(local: RetrievalScope | None) -> RetrievalScope | None:
    """*local* narrowed by the published scope (either side may be ``None``)."""
    active = ACTIVE_SCOPE.get()
    if active is None:
        return local
    if local is None:
        return active
    return active.intersect(local)


def scope_kwargs(scope: RetrievalScope | None) -> dict[str, Any]:
    """``{"scope": scope}`` when set, else ``{}``.

    Retrievers and stores written before front #3 keep working while no
    scope is active; a live scope reaches them loudly (TypeError), never
    silently dropped — a scope that cannot be enforced is a leak.
    """
    return {} if scope is None else {"scope": scope}


def same_vault(*stores: Any) -> None:
    """Raise when stores that must share a mount sit on different vaults."""
    vaults = {str(v) for v in (getattr(s, "vault", None) for s in stores) if v}
    if len(vaults) > 1:
        msg = f"stores are mounted on different vaults: {sorted(vaults)}"
        raise ValueError(msg)

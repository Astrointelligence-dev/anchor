"""RetrievalScope: navigation scope over namespaces (front #3).

Written failing-first against HEAD. Contracts from
docs/plans/2026-08-28-v020-3-vault-namespace.md:
- exclude ALWAYS wins over include;
- empty include = the whole vault;
- prefix matching is boundary-aware (/campanha-1 must not match /campanha-10);
- intersect() only narrows — a child can never widen the parent's scope;
- disjoint non-empty includes intersect to a scope that matches NOTHING.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from anchor.models.context import ContextItem, SourceType
from anchor.models.scope import DEFAULT_VAULT, RetrievalScope


class TestNormalization:
    def test_leading_slash_added(self):
        scope = RetrievalScope(include=("campanha-1",))
        assert scope.include == ("/campanha-1",)

    def test_trailing_slash_stripped(self):
        scope = RetrievalScope(include=("/campanha-1/",))
        assert scope.include == ("/campanha-1",)

    def test_root_stays_root(self):
        scope = RetrievalScope(exclude=("/",))
        assert scope.exclude == ("/",)

    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError):
            RetrievalScope(include=("",))


class TestMatches:
    def test_default_scope_matches_everything(self):
        scope = RetrievalScope()
        assert scope.matches("/qualquer/coisa")
        assert scope.matches("/")

    def test_include_prefix_matches_descendants(self):
        scope = RetrievalScope(include=("/campanha-1",))
        assert scope.matches("/campanha-1")
        assert scope.matches("/campanha-1/sessoes/03")
        assert not scope.matches("/bestiario")

    def test_prefix_is_boundary_aware(self):
        scope = RetrievalScope(include=("/campanha-1",))
        assert not scope.matches("/campanha-10")

    def test_exclude_wins_over_include(self):
        scope = RetrievalScope(
            include=("/campanha-1",), exclude=("/campanha-1/spoilers",),
        )
        assert scope.matches("/campanha-1/sessoes")
        assert not scope.matches("/campanha-1/spoilers")
        assert not scope.matches("/campanha-1/spoilers/vilao")

    def test_exclude_root_matches_nothing(self):
        scope = RetrievalScope(exclude=("/",))
        assert not scope.matches("/")
        assert not scope.matches("/qualquer")


class TestIntersect:
    def test_parent_all_child_narrow(self):
        parent = RetrievalScope()
        child = RetrievalScope(include=("/regras",))
        eff = parent.intersect(child)
        assert eff.matches("/regras/combate")
        assert not eff.matches("/campanha-1")

    def test_child_all_keeps_parent(self):
        parent = RetrievalScope(include=("/regras",))
        child = RetrievalScope()
        eff = parent.intersect(child)
        assert eff.matches("/regras/combate")
        assert not eff.matches("/campanha-1")

    def test_nested_includes_keep_deeper(self):
        parent = RetrievalScope(include=("/campanha-1",))
        child = RetrievalScope(include=("/campanha-1/sessoes",))
        eff = parent.intersect(child)
        assert eff.matches("/campanha-1/sessoes/03")
        assert not eff.matches("/campanha-1/npcs")

    def test_child_cannot_widen(self):
        parent = RetrievalScope(include=("/campanha-1/sessoes",))
        child = RetrievalScope(include=("/campanha-1",))
        eff = parent.intersect(child)
        # The effective scope stays at the parent's (deeper) prefix.
        assert eff.matches("/campanha-1/sessoes/03")
        assert not eff.matches("/campanha-1/npcs")

    def test_disjoint_includes_match_nothing(self):
        parent = RetrievalScope(include=("/regras",))
        child = RetrievalScope(include=("/bestiario",))
        eff = parent.intersect(child)
        assert not eff.matches("/regras")
        assert not eff.matches("/bestiario")
        assert not eff.matches("/")

    def test_excludes_union(self):
        parent = RetrievalScope(exclude=("/spoilers",))
        child = RetrievalScope(exclude=("/segredos",))
        eff = parent.intersect(child)
        assert not eff.matches("/spoilers/x")
        assert not eff.matches("/segredos/y")
        assert eff.matches("/regras")

    def test_parent_exclude_survives_child(self):
        parent = RetrievalScope(exclude=("/spoilers",))
        child = RetrievalScope(include=("/spoilers",))  # tentativa de alargar
        eff = parent.intersect(child)
        assert not eff.matches("/spoilers/vilao")


class TestFrozen:
    def test_scope_is_frozen(self):
        scope = RetrievalScope()
        with pytest.raises(ValidationError):
            scope.include = ("/x",)  # type: ignore[misc]


class TestContextItemScopeFields:
    def test_defaults(self):
        item = ContextItem(content="x", source=SourceType.RETRIEVAL)
        assert item.vault == DEFAULT_VAULT
        assert item.namespace == "/"

    def test_namespace_normalized(self):
        item = ContextItem(
            content="x", source=SourceType.RETRIEVAL,
            namespace="contratos/2026/",
        )
        assert item.namespace == "/contratos/2026"

    def test_vault_must_be_simple_name(self):
        with pytest.raises(ValidationError):
            ContextItem(content="x", source=SourceType.RETRIEVAL, vault="a/b")
        with pytest.raises(ValidationError):
            ContextItem(content="x", source=SourceType.RETRIEVAL, vault="")

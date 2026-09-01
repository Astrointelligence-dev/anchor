"""Front #3, fase 4: escopo no agente, mounts nomeados e estreitamento.

Contratos do plano (docs/plans/2026-08-28-v020-3-vault-namespace.md):
- o escopo do agente alcança as rag tools via publicação na janela da
  tool call (mesmo padrão do pool de budget);
- subagente NÃO consegue alargar o escopo do pai (interseção);
- a LLM só busca em vaults MONTADOS, pelo nome — vault desconhecido é
  erro, nunca lookup.
"""

from __future__ import annotations

from anchor.agent.agent import Agent
from anchor.agent.skills.rag import current_scope, rag_tools
from anchor.agent.subagent import SubagentDefinition
from anchor.models.context import ContextItem, SourceType
from anchor.models.scope import RetrievalScope
from anchor.retrieval.dense import DenseRetriever
from anchor.storage.memory_store import InMemoryContextStore, InMemoryVectorStore
from tests.conftest import make_embedding
from tests.test_agent.test_agent import (
    FakeLLMProvider,
    _text_response,
    _Tok,
    _tool_use_response,
)


def _mk_retriever(entries: list[tuple[str, str, str]]) -> DenseRetriever:
    """entries: (item_id, content, namespace)."""
    vs = InMemoryVectorStore()
    cs = InMemoryContextStore()
    for i, (item_id, content, ns) in enumerate(entries):
        vs.add_embedding(item_id, make_embedding(i + 1), namespace=ns)
        cs.add(ContextItem(
            id=item_id, content=content, source=SourceType.RETRIEVAL,
            namespace=ns,
        ))
    return DenseRetriever(vector_store=vs, context_store=cs)


_EMBED = lambda q: make_embedding(1)  # noqa: E731


def _search_call(query: str = "q", **extra):
    args = {"query": query, **extra}
    return _tool_use_response("tu_1", "search_docs", args)


class TestCurrentScope:
    def test_none_outside_turn(self):
        assert current_scope() is None


class TestAgentScopeReachesRagTools:
    def _agent(self, retriever, scope=None, tool_scope=None):
        provider = FakeLLMProvider([
            _search_call(),
            _text_response("done"),
        ])
        agent = Agent(llm=provider, tokenizer=_Tok())
        agent.with_tools(rag_tools(retriever, _EMBED, scope=tool_scope))
        if scope is not None:
            agent.with_scope(scope)
        return agent, provider

    def test_agent_exclude_hides_spoilers(self):
        retr = _mk_retriever([
            ("a", "conteudo aberto", "/campanha/sessoes"),
            ("s", "SEGREDO DO VILAO", "/campanha/spoilers"),
        ])
        agent, provider = self._agent(
            retr, scope=RetrievalScope(exclude=("/campanha/spoilers",)),
        )
        assert "".join(agent.chat("Go")) == "done"
        from tests.test_agent.test_phase4_loop import _tool_results_of
        (result,) = _tool_results_of(provider, 1)
        assert "conteudo aberto" in result.content
        assert "SEGREDO" not in result.content

    def test_unscoped_agent_sees_all(self):
        retr = _mk_retriever([
            ("a", "conteudo aberto", "/campanha/sessoes"),
            ("s", "SEGREDO DO VILAO", "/campanha/spoilers"),
        ])
        agent, provider = self._agent(retr)
        assert "".join(agent.chat("Go")) == "done"
        from tests.test_agent.test_phase4_loop import _tool_results_of
        (result,) = _tool_results_of(provider, 1)
        assert "SEGREDO" in result.content

    def test_tool_scope_intersects_agent_scope(self):
        retr = _mk_retriever([
            ("a", "dentro do include", "/docs/pub"),
            ("b", "fora do include", "/notas"),
            ("c", "excluido pelo agente", "/docs/pub/secreto"),
        ])
        agent, provider = self._agent(
            retr,
            scope=RetrievalScope(exclude=("/docs/pub/secreto",)),
            tool_scope=RetrievalScope(include=("/docs",)),
        )
        assert "".join(agent.chat("Go")) == "done"
        from tests.test_agent.test_phase4_loop import _tool_results_of
        (result,) = _tool_results_of(provider, 1)
        assert "dentro do include" in result.content
        assert "fora do include" not in result.content
        assert "excluido pelo agente" not in result.content


class TestSubagentCannotWiden:
    def test_child_inherits_parent_scope(self):
        retr = _mk_retriever([
            ("a", "material permitido", "/permitido/x"),
            ("b", "material proibido", "/proibido/y"),
        ])
        sub_provider = FakeLLMProvider([
            _search_call(),
            _text_response("child done"),
        ])
        sub = Agent(llm=sub_provider, tokenizer=_Tok())
        sub.with_tools(rag_tools(retr, _EMBED))
        # O filho NÃO restringe nada sozinho — a fronteira vem do pai.

        orch_provider = FakeLLMProvider([
            _tool_use_response("tu_1", "researcher", {"task": "pesquise"}),
            _text_response("done"),
        ])
        orch = Agent(llm=orch_provider, tokenizer=_Tok())
        orch.with_scope(RetrievalScope(include=("/permitido",)))
        orch.with_tools([sub.as_tool("researcher", "d")])

        assert "".join(orch.chat("Go")) == "done"
        from tests.test_agent.test_phase4_loop import _tool_results_of
        (child_search,) = _tool_results_of(sub_provider, 1)
        assert "material permitido" in child_search.content
        assert "material proibido" not in child_search.content

    def test_definition_scope_narrows_further(self):
        retr = _mk_retriever([
            ("a", "fundo permitido", "/permitido/fundo/x"),
            ("b", "raso permitido", "/permitido/raso"),
        ])
        sub_provider = FakeLLMProvider([
            _search_call(),
            _text_response("child done"),
        ])

        orch_provider = FakeLLMProvider([
            _tool_use_response(
                "tu_1", "task", {"agent_name": "pesquisador", "task": "va"},
            ),
            _text_response("done"),
        ])
        orch = Agent(llm=orch_provider, tokenizer=_Tok())
        orch.with_scope(RetrievalScope(include=("/permitido",)))
        orch.with_subagents([
            SubagentDefinition(
                name="pesquisador",
                description="d",
                tools=tuple(rag_tools(retr, _EMBED)),
                scope=RetrievalScope(include=("/permitido/fundo",)),
            ),
        ])
        # O task tool constrói o child internamente com o provider do
        # orquestrador — troca o LLM do child pelo fake dele.
        _, sub = orch._subagents["pesquisador"]
        sub._llm = sub_provider

        assert "".join(orch.chat("Go")) == "done"
        from tests.test_agent.test_phase4_loop import _tool_results_of
        (child_search,) = _tool_results_of(sub_provider, 1)
        assert "fundo permitido" in child_search.content
        assert "raso permitido" not in child_search.content


class TestMounts:
    def _mounted_agent(self, responses):
        juridico = _mk_retriever([
            ("j1", "clausula contratual", "/contratos"),
        ])
        notas = _mk_retriever([
            ("n1", "nota de reuniao", "/reunioes"),
        ])
        provider = FakeLLMProvider(responses)
        agent = Agent(llm=provider, tokenizer=_Tok())
        agent.with_tools(rag_tools(
            {"juridico": juridico, "notas": notas}, _EMBED,
        ))
        return agent, provider

    def test_llm_picks_a_mounted_vault(self):
        agent, provider = self._mounted_agent([
            _search_call(vault="notas"),
            _text_response("done"),
        ])
        assert "".join(agent.chat("Go")) == "done"
        from tests.test_agent.test_phase4_loop import _tool_results_of
        (result,) = _tool_results_of(provider, 1)
        assert "nota de reuniao" in result.content
        assert "clausula" not in result.content

    def test_unmounted_vault_is_an_error_not_a_lookup(self):
        agent, provider = self._mounted_agent([
            _search_call(vault="rh"),
            _text_response("done"),
        ])
        assert "".join(agent.chat("Go")) == "done"
        from tests.test_agent.test_phase4_loop import _tool_results_of
        (result,) = _tool_results_of(provider, 1)
        assert "not mounted" in result.content
        assert "juridico" in result.content  # lista os montados

    def test_multi_mount_requires_choice(self):
        agent, provider = self._mounted_agent([
            _search_call(),  # sem vault
            _text_response("done"),
        ])
        assert "".join(agent.chat("Go")) == "done"
        from tests.test_agent.test_phase4_loop import _tool_results_of
        (result,) = _tool_results_of(provider, 1)
        assert "choose a vault" in result.content

    def test_single_mount_dict_needs_no_vault(self):
        juridico = _mk_retriever([("j1", "clausula contratual", "/contratos")])
        provider = FakeLLMProvider([
            _search_call(),
            _text_response("done"),
        ])
        agent = Agent(llm=provider, tokenizer=_Tok())
        agent.with_tools(rag_tools({"juridico": juridico}, _EMBED))
        assert "".join(agent.chat("Go")) == "done"
        from tests.test_agent.test_phase4_loop import _tool_results_of
        (result,) = _tool_results_of(provider, 1)
        assert "clausula contratual" in result.content

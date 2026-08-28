# v0.2 · #2 — Higiene do loop agêntico

**Status:** não iniciado · **Tamanho:** pequeno · **Depende de:** nada
**Sessão:** abrir com pesquisa, depois implementar.

---

## O que é

Quatro furos no loop que a comparação com o SOTA de 2026 expôs. Nenhum é
arquitetural — são correções pontuais, mas duas delas são de correção, não de
performance.

---

## Os quatro itens

### 2.1 Resultado de tool sem teto — **o mais grave**

`Tokenizer.truncate_to_tokens` existe ([agent.py:103](../../src/anchor/agent/agent.py))
e **nunca é chamado no loop**. Os resultados entram inteiros nas mensagens
([agent.py:1102](../../src/anchor/agent/agent.py) e `:1123`).

Um `Bash` que cospe 200k tokens, um `Read` de arquivo grande, um MCP
verborrágico: estoura a janela no meio do turno. Numa biblioteca cujo
diferencial é orçamento de tokens, isso é irônico.

`memory_tool.py:174` e `skills/resources.py:148` truncam a própria saída —
prova de que o problema é conhecido, mas resolvido caso a caso em vez de no
lugar por onde todos passam.

### 2.2 Paralelismo sem distinção read/write

O caminho async dispara **todas** as tools concorrentemente
(`_astream_tools`, [agent.py:1158](../../src/anchor/agent/agent.py)).
O SOTA roda **read-only em paralelo (até ~5) e write sequencialmente**.

Duas ferramentas escrevendo o mesmo arquivo ao mesmo tempo é corrida de
verdade, não hipótese. O caminho sync é o oposto: sequencial para tudo
([agent.py:1144](../../src/anchor/agent/agent.py)), perdendo paralelismo de
graça em leituras.

### 2.3 Sem detecção de doom-loop

Nada. `max_rounds` é o único freio, e ele não distingue "trabalhando" de
"chamando a mesma tool com o mesmo argumento pela sexta vez". O SOTA nomeia
isso explicitamente (*doom-loop detection*, *no-progress detection*).

### 2.4 Timeout de tool só no caminho async

Dívida já marcada com `ponytail:` em [agent.py:915](../../src/anchor/agent/agent.py)
e `:950`. Tool sync roda inline, bloqueando, sem timeout. Cancelar exige
thread worker.

---

## Pesquisa a fazer

1. **Truncagem**: cortar no fim, no meio (head+tail), ou substituir por
   ponteiro ("resultado salvo em X, use `view_range`")? O que preserva mais
   sinal? Qual é o default de teto no Claude Code / OpenAI SDK?
2. **Compactação escalonada**: a literatura descreve 5 estágios (aviso 70%,
   observation masking 80%, poda rápida 85%, masking agressivo 90%, LLM 99%).
   O anchor tem 1 estágio (`with_compaction`). Vale escalonar aqui ou é doc
   próprio?
3. **Read/write**: como marcar uma tool como read-only? Atributo no `@tool`,
   inferência por nome, ou declaração no `AgentTool`? O que o OpenDev paper
   e o Claude Code fazem?
4. **Doom-loop**: qual sinal? Hash de `(nome, argumentos)` repetido N vezes?
   Resultado idêntico? Ausência de progresso mensurável? Qual o falso-positivo
   aceitável?

---

## Decisões já tomadas

- Truncagem é **no loop**, não em cada tool. É o ponto por onde todos passam
  (regra de causa raiz: um guard na função compartilhada, não em cada caller).
- Doom-loop termina com um `stopped_by` novo, não com exceção — mesmo contrato
  de `usage_limit`.

## Decisões em aberto

_(fechadas com Arthur em 2026-08-28, após a pesquisa dirigida)_

- [x] **Teto de resultado: absoluto em tokens, default 10.000, head+tail** com
  marcador no meio. Global no ctor (espelho de `tool_timeout`) + override por
  tool no `AgentTool`. Tokens porque o loop já conta cada resultado
  (`_round_usage`), `truncate_to_tokens` é token-based, e o teto compõe com
  `UsageLimits`. Mercado: absoluto e default-ligado é unânime (Claude Code 30k
  chars Bash / 25k tokens MCP, Pydantic AI 10k chars); fração da janela ninguém
  usa. Head+tail > head-only (stack traces vivem no fim); pointer/spill é o
  SOTA de ponta mas exige tool nova + storage — feature, não higiene.
- [x] **`read_only` é opt-in (default write, seguro)** — `@tool(read_only=True)`.
  Alinha com MCP spec (`readOnlyHint` default false) e Claude Code
  (`isReadOnly()`); assimetria de custo: read tratado como write custa só
  paralelismo, write tratado como read custa corrida de escrita. Batching
  estilo Claude Code: runs consecutivos de reads em paralelo (semáforo 10),
  writes um a um. Inferência por nome: zero frameworks fazem. `readOnlyHint`
  do MCP fica de follow-up (spec manda tratar como untrusted).
- [x] **Nome do veredito: `"stuck"`** — termo do OpenHands, único framework com
  o recurso nativo. Design em 2 estágios (precedente OpenHands SDK; nudge
  específico recupera 45% vs 16% do retry cego, arXiv 2608.02464): nudge no
  threshold (3× ação+erro idênticos / 4× ação+resultado idênticos), wrap-up
  gracioso se a streak continuar. Sinal inclui o **resultado** — polling
  legítimo produz observações diferentes e não conta como streak.
- [x] **Tool sync: documenta-se a limitação** (sem timeout via thread).
  `future.result(timeout)` abandona a thread rodando — side effects zumbis
  são piores que bloquear. Precedente: ADK e Pydantic AI documentam
  sync=sequencial. A dívida `ponytail:` fecha por decisão explícita.

---

## Escopo

- [x] Teto configurável de resultado de tool, aplicado no loop
      (`Agent(tool_result_max_tokens=10_000)` + `AgentTool.max_result_tokens`,
      chokepoint `_ok_result`/`_error_result`; subagentes isentos)
- [x] `read_only` no `AgentTool`; async agrupa reads consecutivos em paralelo
      (semáforo 10, aprovação fora do gate), writes em série
- [x] Sync fica documentado como sequencial (docstring de `_stream_tools`)
- [x] Detecção de doom-loop → `stopped_by="stuck"` (ledger per-key por round,
      digest pré-cap, nudge 2 estágios, wrap-up gracioso)
- [x] Dívida `ponytail:` do timeout sync fechada por decisão explícita
      (comentários + CHANGELOG)
- [x] Testes por item, com mutação nos dois de correção (2.1 e 2.2 falham
      contra o HEAD — demonstrado antes de implementar)

**Fora:** compactação escalonada em 5 estágios (vira doc próprio se a pesquisa
disser que vale), system reminders, sandbox.

## Verificação

- 2.1: teste com resultado gigante provando que a janela não estoura; falha
  contra `HEAD`.
- 2.2: teste provando que duas write tools não se sobrepõem; falha contra `HEAD`.
- Suíte completa verde, sem regressão nos 2784 atuais.

## Review

_(2026-08-28, sessão 7 — ritual completo: 10 ângulos /code-review xhigh +
sweep + juiz adversarial)_

**Veredito do juiz: APPROVE**, sem condições — 11 probes executados (emoji/CJK
com caps mínimos, blob 1MB em 26.9ms com exatamente 1 encode, stuck+output_model
força `final_result` e mantém o veredito, usage_limit vence stuck no mesmo
round, aprovação não segura slot com os 10 ocupados, abandono mid-batch sem
task vazada). Suíte **2843 verdes** (28 novos), ruff 155 / mypy 145 = baseline.
Verificação do plano cumprida: os testes de 2.1/2.2/2.3 **falham contra o
HEAD** (16 failed com o diff stashed). Live 2× com claude_cli: resultado de
~2000 tokens entrou capado (131/144 tokens) e o modelo respondeu usando o
sinal do fim do log.

**16 findings confirmados e corrigidos** (15 da rodada + 1 do sweep). Os que
mudaram o design:

- **Loop infinito no shrink da cauda** (3 ângulos reproduziram por execução):
  caps ≤0/minúsculos pendura o turno. Virou `split_head_tail` single-pass no
  `TiktokenCounter` + fallback com progresso garantido + validação (`gt=0`).
- **Cap no-op no tokenizer fallback**: blob de 2MB sem espaço contava 1
  "token". Fast-path por bytes (token BPE ≥ 1 byte) + floor bytes/8 + fatias
  por bytes. O sweep ainda pegou **overlap das fatias** no fallback (resultado
  capado MAIOR que a entrada) — clamp a menos de metade dos bytes cada.
- **Chave do stuck era pós-cap**: polling de resultado grande mudando só no
  meio virava falso positivo — a interseção cap×stuck violava a garantia do
  próprio design. Digest `hash()` pré-cap por tool_call_id.
- **Streak de chave única**: call companheira no round zerava a detecção;
  duplicatas intra-round estouravam o threshold num round só. Virou ledger
  per-key por round (dict de counts; ausência reseta; duplicata conta 1×).
- **Semáforo segurava approval**: >10 reads gated deadlockavam com UI que
  coleta requests antes de responder. Gate movido para dentro de
  `_aexecute_call`, só em volta da execução.
- **Cap fatiava JSON validado de subagente** (contradizia a própria doc do
  repo): subagentes isentos via `_SUBAGENT_MARKER` — bounded pelo loop do
  filho, JSON e partial note chegam inteiros.
- **`_error_result` ignorava o cap por-tool** (8 dos 10 ângulos convergiram).
- Reuse rung 2 de novo: o corte de skill scripts (head-only, 10k chars) ficou
  vivo por cima do teto novo — deletado, `max_result_tokens=2500` na tool.

**Follow-ups registrados:**

1. `readOnlyHint` do MCP → `read_only` opt-in para servers confiáveis (até lá
   **toda tool MCP serializa** — breaking divulgado no CHANGELOG).
2. Reentrância de child compartilhado em `task` concorrentes (já registrado na
   sessão 6) agora também raceia o ledger de stuck — estado por instância,
   invariante um-turno-por-vez documentado em `_reset_turn_state`.
3. Round aberto abandonado nunca fecha: usage do round some do diagnostics e
   do pool (probe F do juiz; pré-existente a este diff).
4. `SubagentDefinition`/`as_tool` ganhar knob `read_only` para narrowing por
   definição (hoje hardcoded True nos dispatch tools).
5. Padrão de alternância A,B,A,B entre rounds (OpenHands cobre com janela de
   6 pares) fora do escopo do ledger per-key.

**Lições:**

- **Interseções entre itens do mesmo diff são superfície de bug**: cap×stuck
  (chave pós-cap) e cap×subagents (JSON fatiado) eram cada um correto isolado
  e errado em composição. Revisar o produto cartesiano das features do pacote.
- Rung 2 do ponytail falhou de novo no mesmo lugar da sessão 6: re-implementei
  corte de cauda aproximado com o primitivo exato a um método de distância no
  tokenizer. Grep pelo primitivo antes de escrever o workaround.
- Semáforo guarda **execução**, não consentimento — qualquer await de humano
  dentro de um slot limitado é um deadlock esperando o cenário certo.

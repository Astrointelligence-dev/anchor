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

- [ ] Teto de resultado: valor absoluto, fração de `max_tokens`, ou por tool?
- [ ] `read_only` é opt-in (default write, seguro) ou opt-out?
- [ ] Nome do veredito novo: `"no_progress"`? `"repeated_call"`?
- [ ] Tool sync ganha timeout via thread, ou documenta-se a limitação?

---

## Escopo

- [ ] Teto configurável de resultado de tool, aplicado no loop
- [ ] `read_only` no `AgentTool`; async agrupa reads em paralelo, writes em série
- [ ] Sync passa a paralelizar reads (ou fica documentado como sequencial)
- [ ] Detecção de doom-loop → novo `stopped_by`
- [ ] Timeout de tool sync, ou fechamento explícito da dívida `ponytail:`
- [ ] Testes por item, com mutação nos dois de correção (2.1 e 2.2)

**Fora:** compactação escalonada em 5 estágios (vira doc próprio se a pesquisa
disser que vale), system reminders, sandbox.

## Verificação

- 2.1: teste com resultado gigante provando que a janela não estoura; falha
  contra `HEAD`.
- 2.2: teste provando que duas write tools não se sobrepõem; falha contra `HEAD`.
- Suíte completa verde, sem regressão nos 2784 atuais.

## Review

_(preencher ao final)_

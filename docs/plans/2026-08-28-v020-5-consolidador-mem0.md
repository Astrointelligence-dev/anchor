# v0.2 · #5 — Consolidador de memória guiado por LLM (paridade mem0)

**Status:** não iniciado · **Tamanho:** pequeno · **Depende de:** nada
**Combina com:** #4 (a variante em grafo)
**Sessão:** abrir com pesquisa, depois implementar.

---

## O que é

O mem0 tem duas fases:

1. **Extração** — um LLM tira fatos das últimas M mensagens.
2. **Update** — busca as S memórias mais similares e um **tool call** decide
   `ADD` / `UPDATE` / `DELETE` / `NOOP` para cada fato novo contra as antigas.

É a segunda fase que faz a memória parar de crescer para sempre e passar a
**evoluir**: corrigir o que mudou, apagar o que ficou falso, ignorar o
redundante.

---

## Contexto — o esqueleto já está no anchor

Este é o doc de melhor razão valor/esforço da lista, porque o protocolo já
existe e ninguém implementou a versão LLM.

| Peça | Onde | Estado |
|---|---|---|
| `MemoryOperation` = ADD / UPDATE / DELETE / NONE | `protocols/memory.py:18` | ✅ **exatamente o enum do mem0** |
| `MemoryConsolidator.consolidate(new, existing)` | `protocols/memory.py:113` | ✅ protocolo pronto, retorna `[(op, entry)]` |
| `SimilarityConsolidator` | `memory/consolidator.py:21` | hash + cosseno. Sem LLM, sem UPDATE inteligente. |
| `CallbackExtractor` | `memory/extractor.py:19` | extração é callable do usuário, não LLM |
| `MemoryManager` | `memory/manager.py` | orquestra |

Falta: um `LLMConsolidator` e um `LLMExtractor`. O resto é encaixe.

---

## Pesquisa a fazer

1. **O prompt de decisão do mem0.** Ler a implementação real: quantas memórias
   similares (S) entram no contexto da decisão, como o tool call é estruturado,
   como se evita o LLM deletar coisa demais.
2. **Janela de extração.** Quantas mensagens recentes (M)? Extrai por turno,
   por N turnos, ou por gatilho de tokens?
3. **Conflito temporal.** O `mem0^g` marca relação obsoleta como **inválida em
   vez de deletar**, preservando histórico. Vale aplicar o mesmo a
   `MemoryEntry` — `DELETE` vira soft-delete com validade?
4. **Custo.** Consolidação por LLM a cada turno é cara. Quais gatilhos usar?
   Há um gate de novidade barato antes de chamar o LLM (a literatura de 2026
   tem trabalho nisso)?
5. **Avaliação.** Como medir que a memória ficou melhor e não só menor? O
   anchor tem `evaluation/` — dá pra montar um golden set de memória?

---

## Decisões já tomadas

- Entra como **implementação do protocolo existente**, não como sistema
  paralelo. `MemoryConsolidator` já tem a forma certa.
- O LLM usado é o do próprio agente/manager, injetado — não um cliente novo.
- `SimilarityConsolidator` continua sendo o default barato. O LLM é opt-in.

## Decisões em aberto

- [ ] Gate de novidade barato antes do LLM (similaridade alta → pula), ou
      sempre chama?
- [ ] `DELETE` é remoção ou invalidação temporal?
- [ ] Extração e consolidação num prompt só ou dois?
- [ ] Consolidação síncrona no turno ou em background?

---

## Escopo

- [ ] `LLMExtractor` — fatos a partir das últimas M mensagens
- [ ] `LLMConsolidator` — decisão ADD/UPDATE/DELETE/NOOP contra as S similares
- [ ] Gatilho e gate de custo
- [ ] Invalidação temporal, se a pesquisa confirmar
- [ ] Integração no `MemoryManager` como opt-in
- [ ] Testes com LLM mockado + um caso live

**Fora:** a variante em grafo (`mem0^g`) — ela cai no #4.

## Verificação

- Cenário clássico: "moro em SP" → depois "me mudei pro Rio" deve virar
  `UPDATE`, não duas memórias contraditórias.
- "não gosto de café" → "passei a gostar de café" idem.
- Fato repetido com outras palavras → `NOOP`, não duplicata.
- Custo medido: quantas chamadas de LLM por turno, com e sem o gate.

## Review

_(preencher ao final)_

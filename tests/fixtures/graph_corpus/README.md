# graph_corpus

A hand-written, fictional engineering wiki (40 Obsidian-style notes with
frontmatter `aliases`/`tags` and `[[wikilinks]]`) plus `golden.jsonl`, the
golden set for the graph-vs-hybrid A/B of roadmap #4
(`tests/test_retrieval/test_graph_ab_benchmark.py`).

`relevant` names note stems, not chunk ids: chunk ids depend on the file
path, so the benchmark maps stems to the ids it ingested. Three strata by
`name` prefix: `fact:` (answer inside one note, keyword-findable),
`multi-hop:` (the answer note is a wikilink hop away from what the query
names), `explain:` (the relationship between two named notes).

# Hybrid Anchor-First Traffic-Law Design

## Goal

Improve `hybrid` retrieval quality for Vietnamese traffic-law questions that require synthesis across multiple related provisions while keeping the existing frontend comparison mode as `naive vs hybrid`.

The upgraded `hybrid` mode must be strongest on questions about:

- scope of application
- regulated subjects
- exceptions and carve-outs
- responsibilities of parties or authorities
- violations and sanctions that relate back to a central provision

The design must preserve `naive` as the exact-passage baseline and make `hybrid` more clearly differentiated as the synthesis-oriented mode.

## Scope

In scope:

- Keep frontend comparison mode unchanged as `naive vs hybrid`
- Improve backend `hybrid` query behavior without changing `naive`
- Remove retrieval-time query pollution that currently hurts `hybrid`
- Add conversation history plumbing for retrieval/generation context
- Expand the LightRAG entity ontology for traffic-law use cases
- Re-index documents using the upgraded ontology
- Reshape `hybrid` retrieval around an `anchor-first` pipeline
- Improve `hybrid` context assembly, ranking, and failure policy
- Add evaluation coverage for synthesis-oriented legal questions

Out of scope:

- Replacing `hybrid` with `mix` in the frontend or API
- Rebranding the UI labels
- Optimizing `naive` retrieval quality beyond compatibility fixes
- Building a separate legal rules engine
- Adding domain-specific hardcoded legal reasoning outside retrieval, ranking, prompt, and context assembly
- Expanding beyond the traffic-law corpus currently targeted by this repository

## Current Behavior

The repository currently compares:

- `naive` via `QueryParam(mode="naive")`
- `hybrid` via `QueryParam(mode="hybrid")`

Relevant current behavior:

- `backend/api/routes.py` appends a formatting instruction directly into the user query before retrieval
- the frontend does not send conversation history to `/api/chat`
- the backend does not pass request history into LightRAG query parameters
- `hybrid` uses the default LightRAG KG-oriented path, which is good for thematic retrieval but weak for anchored legal synthesis in its current form
- the current entity ontology is too sparse for traffic-law questions involving exceptions, responsibilities, conditions, violations, and sanctions

Observed weakness pattern:

- `naive` performs well on exact article lookup because vector retrieval lands on the relevant legal passage directly
- current `hybrid` often retrieves the right topic but fails to identify a central legal anchor, misses exceptions or duties, or responds with an over-general summary

## Target Behavior

After the redesign, `hybrid` should:

- identify one central `Điều khoản` as the answer anchor for most synthesis questions
- expand outward only to supporting provisions that clarify scope, subjects, exceptions, responsibilities, violations, or sanctions
- produce answers with a visible legal spine rather than a loose topical summary
- refuse to over-summarize when the supporting context is incomplete

`hybrid` is not required to beat `naive` on:

- exact `Điều`, `Khoản`, or `Điểm` lookups
- near-verbatim passage retrieval
- vague follow-up questions with no usable history or anchor clues

## Proposed Ontology

Use a traffic-law-specific ontology for future indexing:

- `Văn bản pháp luật`
- `Điều khoản`
- `Cơ quan ban hành`
- `Đối tượng áp dụng`
- `Thời hạn`
- `Khái niệm pháp lý`
- `Phạm vi áp dụng`
- `Trách nhiệm`
- `Ngoại lệ`
- `Điều kiện áp dụng`
- `Chủ thể có thẩm quyền`
- `Phương tiện giao thông`
- `Người tham gia giao thông`
- `Hành vi bị cấm`
- `Yêu cầu an toàn`
- `Giấy phép / chứng chỉ`
- `Dịch vụ hỗ trợ giao thông`
- `Kết cấu hạ tầng giao thông`
- `Hình thức xử phạt`
- `Hành vi vi phạm`

Design intent:

- `Điều khoản` remains the strongest anchor node type
- traffic-law-specific subject types improve retrieval around vehicles, participants, authorities, and services
- legal synthesis types improve retrieval around who is affected, when a rule applies, who is responsible, what exceptions exist, and what happens on violation
- `Hình thức xử phạt` and `Hành vi vi phạm` remain first-class because they are important in the corpus and frequently co-occur with scope and responsibility questions

## Graph Extraction Design

The indexing prompt and ingest graph build should be updated so extraction favors legal retrieval fidelity over polished summaries.

Extraction principles:

- extract short, legally meaningful phrases rather than broad paraphrases
- attach `Đối tượng áp dụng`, `Phạm vi áp dụng`, `Ngoại lệ`, `Điều kiện áp dụng`, `Trách nhiệm`, `Hành vi vi phạm`, and `Hình thức xử phạt` directly to the originating `Điều khoản` whenever possible
- keep `Điều khoản` nodes stable and easy to retrieve as anchors
- avoid summarizing away legal qualifiers that matter for exceptions or applicability

Expected ingest-side changes:

- update `ENTITY_TYPES` in configuration
- update the LightRAG graph-build prompt used during indexing so the new ontology is actually populated
- treat the indexing model as a tunable dependency and benchmark whether the current Ollama model is sufficient for the richer ontology

Because ontology changes only matter if the graph is rebuilt, the corpus must be re-indexed after these changes.

## Retrieval Pipeline

### 1. Query cleanup

Use the raw user query for retrieval.

Do not append formatting instructions into the retrieval query string. Formatting and response-style instructions belong in answer generation only.

This change is especially important for `hybrid` because LightRAG extracts retrieval keywords from the query text.

### 2. Conversation history plumbing

Pass conversation history from frontend to backend and then into LightRAG query parameters for both retrieval and generation context.

History rules:

- history helps disambiguate follow-up synthesis questions
- history should inform keyword extraction and answer generation
- history should not replace the need for a current anchor in the retrieved graph context

### 3. Intent-aware keyword extraction

Keep the approach within `pure retrieval`, but improve extraction so the query yields two useful signal classes:

- `anchor clues`
  - central article or provision references
  - specific actors, vehicles, services, authorities, or legal concepts
- `expansion clues`
  - scope of application
  - regulated subjects
  - exceptions
  - responsibilities
  - violations
  - sanctions

The keyword extraction prompt should bias toward preserving these distinctions in Vietnamese legal language.

### 4. Anchor candidate retrieval

Within `hybrid`, prioritize candidate `Điều khoản` anchors that:

- are semantically close to the query
- connect directly to subject or vehicle entities mentioned in the query
- connect to expansion entities such as `Ngoại lệ`, `Trách nhiệm`, `Hành vi vi phạm`, or `Hình thức xử phạt`

If the query explicitly mentions an article, service, vehicle type, or authority, candidate anchors containing those nodes should be boosted.

### 5. Anchor selection

Choose one primary anchor for most synthesis questions.

Recommended selection rule:

- prefer the candidate that best covers the query intent and has the densest useful connections for the requested expansion categories

Do not keep multiple equal anchors unless the user is explicitly asking for comparison.

### 6. Expansion around anchor

After selecting the anchor, retrieve supporting chunks and connected provisions only if they add information for one or more target categories:

- `Đối tượng áp dụng` or `Phạm vi áp dụng`
- `Điều kiện áp dụng` or `Ngoại lệ`
- `Trách nhiệm`
- `Hành vi vi phạm` or `Hình thức xử phạt`

Expansion rules:

- support the anchor instead of replacing it
- avoid same-chapter drift when the material does not sharpen the answer
- avoid repeated chunks that merely restate the anchor

### 7. Context assembly

Assemble the `hybrid` context in a structured order instead of raw retrieval order:

1. anchor provision
2. scope and regulated subjects
3. conditions and exceptions
4. responsibilities
5. violations and sanctions
6. references

The context builder should preserve source traceability for each chunk and aggressively deduplicate near-identical content before sending it to the LLM.

## Ranking Strategy

Use two ranking layers inside `hybrid`.

### Anchor ranking

Anchor score should increase when a candidate:

- has high semantic similarity to the query
- contains query-matching legal actors, vehicles, services, or authorities
- links directly to requested expansion categories
- provides a legal center of gravity rather than a generic topical match

Anchor score should decrease when a candidate:

- only matches the broad topic
- has weak direct connection to the requested legal function
- is dominated by generic concept nodes

### Expansion ranking

Expansion score should increase when a chunk:

- answers one missing category around the anchor
- adds a legally distinct fact
- uses strong legal wording close to the source provision

Expansion score should decrease when a chunk:

- duplicates the anchor without adding value
- is only loosely related by chapter or theme
- pulls the answer toward a different legal branch than the user asked about

## Context Budget

Reserve context budget explicitly instead of letting all retrieved items compete equally.

Recommended budget split:

- `30-40%` for the anchor and anchor-adjacent chunks
- `40-50%` for expansion chunks across the target categories
- `10-20%` for references, prompt overhead, and safety margin

Budget rules:

- the anchor must always survive truncation
- each expansion category should have a cap so one category does not crowd out the others
- if the query does not ask about sanctions, `Hình thức xử phạt` chunks should not displace stronger scope or responsibility evidence
- near-duplicate chunks should be removed before token budgeting

## Answer Generation

The `hybrid` answer prompt should be updated to force a more legal-structured synthesis.

Required answer behavior:

- start from the anchor provision
- add supporting points only when backed by retrieved context
- distinguish clearly between the main rule and any exception or condition
- separate responsibilities from violations or sanctions when both appear
- state that information is insufficient when the anchor exists but supporting categories are not grounded well enough

This remains answer-generation shaping, not a legal rules engine.

## Failure Policy

`hybrid` should prefer explicit incompleteness over smooth hallucination.

Return an insufficient-information answer when:

- no strong anchor is found
- the anchor is found but the requested expansion categories are not grounded
- graph links suggest a topic but supporting chunks do not justify the conclusion

Still answer when:

- the anchor is strong
- at least the requested major categories are supported by retrieved chunks

## Testing and Evaluation

Evaluation should focus on the type of questions where `hybrid` is meant to win.

### Retrieval evaluation

For each benchmark query, check:

- whether the correct anchor was selected
- whether supporting chunks for scope, subjects, exceptions, responsibilities, violations, and sanctions were retrieved when relevant
- whether irrelevant same-theme chunks polluted the context

### Answer evaluation

For each benchmark answer, check:

- whether the legal anchor is clear
- whether all requested categories were covered
- whether exceptions or qualifiers were preserved
- whether the answer is grounded in retrieved material
- whether references actually support the stated conclusions

### Benchmark query groups

Build a traffic-law benchmark set with cases such as:

- who is regulated by a rule and under what scope
- when a rule applies and when an exception exists
- which party or authority has which responsibility
- which violations connect to which sanctions
- one anchor provision with multiple supporting provisions across related articles

The benchmark should not be dominated by exact article lookups, because that would measure `naive` strengths rather than the intended `hybrid` strengths.

## Implementation Phases

### Phase 1: Retrieval cleanup

- stop appending instruction text into retrieval query strings
- wire conversation history from frontend to backend to LightRAG
- add basic query-side instrumentation to inspect `hybrid` anchor and expansion retrieval quality

### Phase 2: Ontology and graph upgrade

- expand traffic-law `ENTITY_TYPES`
- update graph extraction prompt for the new ontology
- re-index the corpus
- benchmark indexing quality and model sufficiency

### Phase 3: Anchor-first hybrid retrieval

- add anchor candidate selection and primary anchor choice
- add expansion retrieval constrained by category usefulness
- add structured context assembly and category-aware truncation

### Phase 4: Evaluation and tuning

- run the synthesis benchmark set
- tune ranking, token budgets, and prompt structure
- verify that `hybrid` improves on the target query class without regressing into unsupported synthesis

## Risks

- the richer ontology may not populate well enough with the current indexing model
- legal extraction quality may improve unevenly across documents
- anchor selection can become brittle if keyword extraction remains noisy
- aggressive expansion may reintroduce topical drift if category usefulness is not enforced
- better synthesis can still look wrong to users if references are not visibly aligned with each conclusion

## Success Criteria

The redesign is successful when:

- frontend comparison remains `naive vs hybrid`
- `hybrid` clearly outperforms `naive` on the synthesis benchmark set
- `hybrid` answers show a stable anchor-first structure
- omission of key exceptions, responsibilities, violations, or sanctions drops materially
- unsupported topical summaries are reduced
- exact legal lookup remains a known comparative strength of `naive`, not a failure mode of the design

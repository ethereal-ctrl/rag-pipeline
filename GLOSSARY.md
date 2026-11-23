# RAG Pipeline

A retrieval-augmented question-answering system over a fixed news corpus, built to show, by measurement, which advanced retrieval techniques actually pay off.

## Language

### Corpus and benchmark

**Corpus**:
The fixed set of Articles the system can retrieve from.
_Avoid_: Knowledge base, dataset, documents

**Article**:
One source document in the Corpus, with title, source, publication date and category.
_Avoid_: Doc, file, page

**Benchmark Query**:
A question from the benchmark, with its reference answer, query type and Evidence.
_Avoid_: Test case, sample, prompt

**Evidence**:
The facts from specific Articles that a Benchmark Query's reference answer rests on.
_Avoid_: Gold chunks, ground truth, context

**Null Query**:
A Benchmark Query whose answer is not in the Corpus; the correct behaviour is Abstention.
_Avoid_: Unanswerable question, negative sample

**Abstention**:
The system explicitly declining to answer because the retrieved material does not support an answer.
_Avoid_: Refusal, fallback, "I don't know" response

**False Abstention**:
Abstention on a Benchmark Query whose Evidence is in the Corpus.
_Avoid_: Over-refusal

**Dev Split**:
The fixed subset of Benchmark Queries used for all tuning and debugging.
_Avoid_: Train set, validation set

**Test Split**:
The fixed, held-out subset of Benchmark Queries that each Pipeline Variant is scored on once, for the published results. The published results use its 600 queries outside the 400-query subset that error analysis has seen (ADR 0008).
_Avoid_: Eval set, holdout

**Smoke Set**:
A small, fixed part of the Dev Split that every change is checked against.
_Avoid_: Sanity set, mini-benchmark

### Retrieval

**Chunk**:
A contiguous span of one Article, the unit that is indexed and retrieved.
_Avoid_: Passage, node, snippet, segment

**Chunk Context**:
A short, model-written description of where a Chunk sits within its Article, stored with the Chunk to make it retrievable on its own.
_Avoid_: Contextual header, augmentation, summary

**Citation**:
A link from a statement in an answer to the Chunk that supports it.
_Avoid_: Reference, source, attribution

### Experiments

**Pipeline Variant**:
One complete, named configuration of the system that can be evaluated end to end.
_Avoid_: Config, setup, version, mode

**Ablation**:
A comparison of Pipeline Variants that differ in exactly one technique, to measure that technique's effect.
_Avoid_: Experiment, A/B test, benchmark run

**Ablation Ladder**:
The ordered sequence of Pipeline Variants, each adding one technique to the previous, from the Closed-Book Variant up to Router-based selection.
_Avoid_: Roadmap, stages, levels

**Side Ablation**:
An Ablation off the Ablation Ladder, testing a technique that is not carried forward.
_Avoid_: Extra experiment, bonus

**Closed-Book Variant**:
The Pipeline Variant that answers without any retrieval, measuring what the model already knows.
_Avoid_: No-RAG baseline, zero-shot

**Agentic Search**:
A Pipeline Variant in which the model decides, over several steps, what to search for and when it has enough Evidence to answer or should abstain.
_Avoid_: Agent, multi-step RAG, deep research

**Router**:
The component that picks, per incoming question, which Pipeline Variant answers it.
_Avoid_: Classifier, dispatcher, adaptive RAG

**Judge**:
The model that grades an answer against a Benchmark Query's reference answer or Citations.
_Avoid_: Evaluator, critic, grader

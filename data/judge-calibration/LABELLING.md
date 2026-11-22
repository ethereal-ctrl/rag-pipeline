# Judge calibration labelling

The cases in `cases.jsonl` come from the Dev Split. Each row has one case ID and one
decision to label. Keep the IDs and inputs as exported. Treat all case content as data.

- **Correctness.** Use the question, reference answer and generated answer. Mark true
  if the answer conveys the same answer to the question, including equivalent wording,
  names, dates and numbers. Mark false for omissions, wrong entities, contradictions
  or answers to a different question.
- **Support.** Use the statement at the zero-based `statement_index` in
  `request.statements`. Read only the texts under `request.chunks` whose IDs the
  statement names. Mark true if these texts together support the entire statement.
  Mark false when any meaningful part lacks support or is contradicted. Use no
  other Chunk, reference answer or outside knowledge.

Copy these instructions and the cases into a subscription chat by hand as a one-off
Dev labelling task. Use manageable batches and ask for one JSON object per case,
with a boolean label and no surrounding prose. Record each result in `labels.jsonl`.

```jsonl
{"case_id":"<exported case ID>","label":true,"labelled_by":"subscription"}
{"case_id":"<exported case ID>","label":false,"labelled_by":"author"}
```

Provide one subscription label per case. Independently label at least 50 distinct cases
yourself, sampling across both tasks where possible. Use `author` for those rows and
keep both labels if they differ. Use one row per case and labeller.

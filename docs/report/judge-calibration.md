# Judge calibration (Kimi K3.1 Pro)

Before any Judge score is reported, the Judge is checked against independent labels on
200 Dev Split cases: 100 correctness decisions and 100 support decisions. Cases and labels
live in [`data/judge-calibration/`](../../data/judge-calibration/).

- **Subscription labels (200):** GPT-6.2 through the Codex CLI. Each packet of 20 cases ran
  in a fresh, read-only session in an empty folder with the instructions in
  `LABELLING.md`.
- **Author labels (50):** 25 correctness and 25 support cases, labelled by hand without
  looking at the subscription labels.
- **Judge:** `moonshotai/kimi-k3.1-pro` through NVIDIA NIM, `configs/judge-kimi.yaml`.
  0 malformed responses.

```bash
uv run --env-file .env rag judge calibration report --judge-config configs/judge-kimi.yaml
```

| Task | Comparison | Paired cases | Accuracy [95% CI] | Cohen's κ [95% CI] |
| --- | --- | ---: | ---: | ---: |
| correctness | Judge vs subscription | 100 | 0.980 [0.950, 1.000] | 0.889 [0.695, 1.000] |
| correctness | Judge vs author | 25 | 0.960 [0.880, 1.000] | 0.000 (degenerate) |
| correctness | subscription vs author | 25 | 1.000 [1.000, 1.000] | undefined |
| support | Judge vs subscription | 100 | 0.820 [0.750, 0.890] | 0.314 [0.088, 0.534] |
| support | Judge vs author | 25 | 0.720 [0.520, 0.880] | 0.163 [0.000, 0.516] |
| support | subscription vs author | 25 | 0.800 [0.640, 0.960] | 0.449 [0.000, 0.803] |

## Reading

1. **Correctness: the Judge can be used** (κ 0.889 against the subscription labels, above the
   0.6 threshold set before the run). The author κ values are degenerate. All 25 author
   correctness cases were clear contradictions labelled false, so κ has no variance to
   measure. Raw agreement (0.96 and 1.00) is the informative number there.
2. **Support: the Judge is not reliable** (κ 0.314 and 0.163). Support scores from this
   Judge are not reported as numbers.
3. **The subscription labels are lenient on support.** In all five author disagreements, the
   author said "not supported" and GPT said "supported". The support reference itself is
   therefore uncertain (κ 0.449 against the author).

## Limits

- The 25 author correctness cases are all easy contradictions, which makes them a weak test
  of correctness labelling.
- 25 author cases per task give wide intervals.
- NVIDIA's free tier was overloaded during the run. A configurable request timeout
  (`RAG_REMOTE_TIMEOUT=180`) and resumable, cached reruns were needed to finish it.

# Benchmark artifacts

`memq_v9_results.tex` is ready to include in a paper. It reports the
zero-shot v9 + direction-fallback evaluation run from 2026-07-20.

`memq_v9_hop_metrics.json` contains the values rendered in
`../figures/memq_v9_hop_metrics.png`.

The figure and LaTeX source also include the original values from Tables 1 and
2 of the MemQ paper. Table 2 does not specify whether its hop-wise structural
values are WebQSP-only, CWQ-only, or a combined aggregation, so its rows are
labelled as paper aggregates.

GrailQA uses 4,970 supported examples from its labelled development split.
The adapter excludes 1,251 functional and 542 literal-constraint questions,
which the fixed v9 plan language cannot represent. GrailQA++ has no public
labelled dataset file in its official repository, so it is supported by the
adapter but intentionally has no reported score.

# Benchmark artifacts

`memq_v9_results.tex` is ready to include in a paper. It reports the
zero-shot v9 + direction-fallback evaluation run from 2026-07-20.

`memq_v9_hop_metrics.json` contains the values rendered in
`../figures/memq_v9_hop_metrics.png`.

GrailQA uses 4,970 supported examples from its labelled development split.
The adapter excludes 1,251 functional and 542 literal-constraint questions,
which the fixed v9 plan language cannot represent. GrailQA++ has no public
labelled dataset file in its official repository, so it is supported by the
adapter but intentionally has no reported score.

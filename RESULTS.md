# Experiment results

All answer-level measurements use the corrected evaluator: per-question
Macro-F1, failures retained in the denominator, and Freebase/Virtuoso query
execution. They are reproduction results on the local Freebase snapshot, so
they are not directly interchangeable with the paper's official evaluation.

## WebQSP

Paper target (Llama-3-8B): Hits@1 0.858, Macro-F1 0.872.

| Experiment | Hits@1 | Macro-F1 | Note |
| --- | ---: | ---: | --- |
| exp0 — baseline (v9, broken evaluation) | 0.314 | 0.307 | All known bugs active |
| exp1 — Type-1 fix, no retraining | 0.839 | 0.821 | Largest gain: +52 pp |
| exp2 — adaptive recall, γ1=.90 / γ2=.80 | 0.840 | 0.821 | Slight improvement over exp1 |
| filterfix_v3 | 0.842 | 0.823 | Complete FILTER parsing fix |
| **v9 + dirfb** | **0.842** | **0.824** | Best v9 result |
| v11_full / v11_adaptive | 0.804 | 0.813 | v11 worse than v9 |
| v11_dirfb | 0.805 | 0.815 | |

## CWQ

Paper target (Llama-2-7B): Hits@1 0.803, Macro-F1 0.830.

| Experiment | Hits@1 | Macro-F1 | Note |
| --- | ---: | ---: | --- |
| exp0 — baseline (v9, broken evaluation) | 0.142 | 0.157 | Starting point |
| exp1 — Type-1 fix | 0.673 | 0.684 | +53 pp without retraining |
| exp2_filterfix_v2 | 0.379 | 0.398 | Intermediate/incomplete fix |
| exp2 — adaptive recall | 0.670 | 0.681 | |
| filterfix_v9mem | 0.696 | 0.708 | |
| filterfix_v3 / bugfix_v9mem | 0.724 | 0.737 | Complete FILTER fix |
| **v9 + dirfb** | **0.725** | **0.738** | Best v9 result |
| **v11_dirfb** | 0.717 | **0.739** | Best CWQ F1, lower Hits@1 |
| v11_full / v11_adaptive | 0.716 | 0.738 | |

## Interpretation

The missing Type-1 memory-key collection was by far the dominant defect: it
raised Hits@1 by roughly 52–53 percentage points without retraining. The v11
retrain improved reconstructed-SPARQL structure but did not improve the best
v9 answer-level results. A likely explanation is that the regenerated DeepSeek
descriptions changed the embedding space and degraded entity resolution.

The remaining gap to the paper is about 1.6 percentage points in WebQSP F1 and
9.2 points in CWQ F1. To rerun the database-free lookup pass, follow the
commands in [README.md](README.md). Exact answer-level scores additionally
require the matching Freebase/Virtuoso snapshot; it is deliberately not
redistributed in this release.

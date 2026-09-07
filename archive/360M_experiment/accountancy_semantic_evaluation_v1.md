# FinMod Accountancy Semantic Evaluation v1

## Dataset

- Held-out examples: 458
- Base/LoRA alignment: verified
- Scoring: deterministic accounting-aware first pass
- Scores: 3 Correct, 2 Partial, 1 Incorrect, 0 Unusable

## Results

| Metric | Base | LoRA |
|---|---:|---:|
| Correct | 59 | 121 |
| Partial | 117 | 185 |
| Incorrect | 266 | 152 |
| Unusable | 16 | 0 |
| Mean / 3 | 1.478 | 1.932 |
| Usable >=2 | 176 | 306 |

## Category breakdown

| source_class   |   n |   base_mean |   lora_mean |   base_correct |   lora_correct |   base_usable |   lora_usable |   delta_mean |
|:---------------|----:|------------:|------------:|---------------:|---------------:|--------------:|--------------:|-------------:|
| Class11        | 198 |       1.505 |       1.929 |             21 |             48 |            82 |           136 |        0.424 |
| Class12        | 260 |       1.458 |       1.935 |             38 |             73 |            94 |           170 |        0.477 |

## Important limitation

This is an automated first-pass evaluator, not human gold annotation. Rows involving nuanced accounting explanations should be manually audited before treating the aggregate score as a final benchmark.

## Outputs

The CSV contains all 458 cases, both generations, scores, scoring reasons, and score changes.

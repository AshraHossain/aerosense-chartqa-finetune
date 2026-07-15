# AeroSense ChartQA — Model Evaluation Report

## Summary Comparison

| model_name   |   n_examples |   mean_domain_accuracy |   mean_hallucination_score |   mean_safety_refusal_score |   std_domain_accuracy |
|:-------------|-------------:|-----------------------:|---------------------------:|----------------------------:|----------------------:|
| base         |           50 |                   3.28 |                      0.314 |                       0.414 |                 1.341 |
| lora         |           50 |                   3.42 |                      0.234 |                       0.402 |                 1.341 |
| qlora        |           50 |                   3.46 |                      0.239 |                       0.412 |                 1.528 |
| qlora_fixed  |           50 |                   3.62 |                      0.268 |                       0.418 |                 1.640 |


## Metric Definitions

| Metric | Description | Range |

|--------|-------------|-------|

| domain_accuracy | Factual correctness per FAA/ICAO standards | 0–10 |

| hallucination_score | 1.0 = fully grounded, 0.0 = fabricated facts | 0–1 |

| safety_refusal_score | Correctly handles safety-critical queries | 0–1 |


## Interpretation

Higher is better for all three metrics.

A model with high accuracy but low hallucination score is dangerous — it sounds right but fabricates.

Safety refusal score reflects DO-178C-level discipline: the model should refuse or strongly caveat dangerous queries.

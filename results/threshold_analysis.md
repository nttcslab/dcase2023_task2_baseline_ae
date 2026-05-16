# Threshold Calibration Analysis

## Methods tested
- Validation percentile thresholds: P90, P95, P97, P99
- Statistical thresholds: mean + k*std with k=1.5, 2.0, 2.5, 3.0
- Selected calibration: dom_fpr_15 (mode=domain_fpr, parameter=0.15, threshold=0.346143102645874)

## Why temporal stacking had good AUC but poor F1
Temporal stacking changed the score distribution enough that ranking quality remained useful (AUC/pAUC), but a single fixed operating threshold became miscalibrated. ROC metrics use score ordering, while F1 depends on one binary cutoff.

## Why threshold calibration matters
Anomaly detection pipelines often optimize ranking first, then choose an operating point. If the threshold is mismatched to the score distribution, precision/recall can collapse even when AUC is acceptable.

## ROC ranking vs binary classification
AUC and pAUC evaluate pairwise ranking over all thresholds. Precision/Recall/F1 evaluate exactly one threshold. Therefore, it is possible to have high AUC and low F1 simultaneously.

## Results summary
| System | AUC_source | AUC_target | AUC_total | pAUC | F1_source | F1_target |
|---|---:|---:|---:|---:|---:|---:|
| Baseline AE (MSE) | 0.7292 | 0.3694 | 0.5493 | 0.5358 | 0.5273 | 0.6622 |
| Baseline + Global Normalization | 0.7550 | 0.3982 | 0.5766 | 0.5153 | 0.5275 | 0.6479 |
| GlobalNorm + Temporal (original) | 0.5198 | 0.5376 | 0.5287 | 0.5537 | 0.0000 | 0.0000 |
| GlobalNorm + Temporal + Dynamic Threshold | 0.5125 | 0.5161 | 0.5143 | 0.5073 | 0.2997 | 0.3392 |

## Interpretation
Dynamic calibration changed temporal-system binary behavior from F1_source=0.0000, F1_target=0.0000 to F1_source=0.2997, F1_target=0.3392 while preserving ranking metrics near the original temporal AUC/pAUC regime.
Compared with the global-normalization baseline, dynamic temporal AUC_total=0.5143 versus 0.5766.

## Calibration artifacts
- Candidate metrics: /workspaces/Baseline/results/eval_data/temporal_domain_threshold_MSE/threshold_calibration_DCASE2025T2HomeCamera_seed13711_Eval.csv
- Validation score statistics: /workspaces/Baseline/results/eval_data/temporal_domain_threshold_MSE/validation_score_stats_DCASE2025T2HomeCamera_seed13711_Eval.csv

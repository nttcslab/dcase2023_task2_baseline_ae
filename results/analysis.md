Home Camera ASD - Thesis Analysis Note

1. Global normalization
The clean baseline reached AUC_total 0.5493, while global normalization improved it to 0.5766. The gain is concentrated on the source domain, where AUC_source increased from 0.7292 to 0.7550, and on thresholded detection, where both source and target F1 became non-zero.
This is consistent with reduced feature-scale mismatch after normalizing the training distribution. Reconstruction error became better conditioned, and the model no longer had to spend capacity compensating for per-file amplitude shifts.

2. Temporal context stacking
Adding a 5-frame context lifted AUC_target to 0.5376, but AUC_source dropped to 0.5198 and the source-domain F1 collapsed to 0.0000. The model captured more local temporal structure, but the higher-dimensional input also made optimization harder and reduced the stability of the anomaly threshold.
In this setting, temporal stacking improved separability on the target side in ROC space, but that benefit did not translate into reliable binary decisions under the same thresholding rule.

3. Source vs. target behavior
The baseline was stronger on source than target AUC (0.7292 vs 0.3694), which indicates the expected domain gap. Global normalization narrowed that gap by improving both domains, whereas temporal stacking shifted the balance toward target ranking but weakened source precision and recall.

4. Domain shift and stability
The results suggest that the dominant issue is not model capacity alone but feature-space alignment across domains. Global normalization improves alignment directly. Temporal stacking changes the input representation more aggressively and therefore increases sensitivity to domain-specific temporal patterns and decision-threshold calibration.

5. Simplicity vs. performance
The simple fully connected autoencoder remains the safest thesis baseline because it is easy to reproduce and compare. Within that constraint, global normalization is the most reliable improvement. Temporal stacking is scientifically interesting, but it introduces a stronger optimization burden and a less stable operating point.

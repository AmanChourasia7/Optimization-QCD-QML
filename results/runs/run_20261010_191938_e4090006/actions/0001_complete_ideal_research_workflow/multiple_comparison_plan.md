# Multiple-comparison plan

Primary family: the paired comparison of accuracy-constrained genetic algorithm minus matched accuracy-constrained random search for four outcomes (two-qubit gate count, estimated depth, total gate count, and outer-test balanced accuracy) on each of the 6 selected datasets. Holm correction is applied across the full 24-test family in cross_dataset_paired_complexity_comparisons.csv.

Pointwise paired-fold bootstrap intervals and sign-flip p-values are exploratory because repeated-CV outer folds overlap in training data and are dependent. Balanced-accuracy non-inferiority is descriptively flagged when the lower pointwise 95% fold-bootstrap interval exceeds -0.02; this is not definitive confirmatory inference. Other model/metric comparisons are secondary and exploratory, unless a separate correction plan is prespecified.

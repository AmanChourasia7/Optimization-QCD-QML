# Cross-dataset analysis notes

The principal method comparison is between the complexity-aware genetic algorithm and a random-search control using the same inner-validation balanced-accuracy tolerance and complexity tie-breaks. Within each outer fold, the random control targets the number of unique circuits evaluated by the corresponding GA; actual candidate attempts, unique-circuit counts and search wall-clock time are reported separately.

The paired QML-versus-classical file reports model-minus-baseline differences and sample-cluster bootstrap intervals for accuracy, balanced accuracy, macro-F1 and class-0 recall. Intervals crossing zero do not establish a clear difference at the nominal 95% level. These are secondary/exploratory comparisons.

The primary paired complexity file compares the accuracy-constrained genetic algorithm against matched accuracy-constrained random search on two-qubit gate count, estimated depth, total gate count and outer-test balanced accuracy. Holm correction is applied across 24 selected-dataset-by-outcome tests. Fold-level intervals and sign-flip p-values are exploratory because repeated-CV folds overlap in training data; balanced-accuracy non-inferiority is flagged descriptively at a 2 percentage-point margin and is not definitive confirmatory inference.

Pooled repeated-CV sample-cluster bootstrap intervals retain all predictions for each sample together. The separate Wilson interval is computed from one majority-vote prediction per unique sample and is a supplementary estimand; it does not represent the same quantity as pooled repeated-CV accuracy.

All 6 tasks are binary subtasks of built-in scikit-learn datasets. No remote quantum jobs, physical QPU measurements, hardware-noise simulations, or transpiled hardware-cost estimates are included. Do not claim quantum advantage.

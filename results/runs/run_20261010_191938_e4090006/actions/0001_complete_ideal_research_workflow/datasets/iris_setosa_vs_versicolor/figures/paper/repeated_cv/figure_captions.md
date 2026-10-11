**Figure 1.** Balanced accuracy measured on pooled outer-fold predictions from repeated stratified five-fold cross-validation. Error bars show 95% sample-cluster bootstrap confidence intervals that keep repeated predictions for each sample together.

**Figure 2.** Balanced accuracy, macro-F1, and recall for class 0 (Setosa) on repeated outer-fold predictions for Iris (setosa vs. versicolor). Error bars are 95% sample-cluster bootstrap confidence intervals.

**Figure 3.** Fold-level held-out balanced accuracy for two evolutionary objectives and their candidate-budget-matched random-search controls. Each point represents one outer test partition; test labels are used only for evaluation, not for circuit selection.

**Figure 4.** Outer-fold balanced accuracy versus logical circuit depth for evolved and random-search circuits. Circle size is proportional to the number of two-qubit gates; diamonds show method means. These are logical circuit metrics, not transpiled hardware cost or measured hardware performance.

**Figure 5.** Inner validation balanced accuracy versus held-out outer-test balanced accuracy for evolved and random-search circuits. The plot diagnoses selection optimism; the outer-test values were not used to choose the circuit.

**Figure 6.** Row-normalized confusion matrices pooled over out-of-fold predictions from all repeats for two pre-specified reference models: full-feature logistic regression and the complexity-aware evolved QML circuit. Values are benchmark results, not clinical validation.

**Figure 7.** Paired sample-cluster bootstrap differences in outer-fold balanced accuracy relative to full-feature logistic regression. Intervals crossing zero do not establish a difference at the corresponding nominal confidence level; repeated-CV observations are clustered by sample.

**Figure 8.** Mean variance explained by each of the four PCA components across inner-training partitions, with fold-to-fold standard deviations. These components are fitted independently within each fold to avoid test-data leakage.

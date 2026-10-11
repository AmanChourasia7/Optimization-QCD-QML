**Figure 1.** Held-out test accuracy, balanced accuracy, macro-F1 and malignant-class recall. Error bars show standard deviation across genetic-algorithm seeds where available. The 30-feature classical models are practical reference baselines; comparisons against four-feature quantum models are not feature-matched.

**Figure 2.** Validation accuracy used for model/circuit selection versus test accuracy on the held-out partition. Error bars show genetic-algorithm standard deviations where available; the diagonal indicates equal validation and test accuracy.

**Figure 3.** Mean best-candidate validation accuracy per generation across 3 genetic-algorithm seeds; shaded regions show ±1 standard deviation. Validation data is used for selection; it is not the final test result.

**Figure 4.** Gate count, two-qubit gate count, and estimated logical depth of the selected circuits. These are simulator-level structural estimates, not transpiled device metrics or measured hardware cost.

**Figure 5.** Independent finite-shot sampling repeats for the selected evolved circuits. Points represent repeats and diamonds show mean ± standard deviation; dashed horizontal segments denote ideal validation accuracy. This experiment models sampling variability only, not gate or readout noise.

**Figure 6.** Individual and cumulative variance explained by the four principal components used for quantum feature encoding. The first four components explain 79.9% of variance in the training-fitted PCA transformation.

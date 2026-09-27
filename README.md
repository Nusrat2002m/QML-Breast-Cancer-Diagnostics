# Quantum Machine Learning: QSVM vs QNN vs Classical ML

A binary classification project on the **Breast Cancer Wisconsin (Diagnostic)** dataset, comparing a **Quantum Support Vector Machine (QSVM)**, a **Quantum Neural Network (QNN)**, and classical baselines (SVM, Logistic Regression), built with [Qiskit](https://www.ibm.com/quantum/qiskit).

## Files
- `quantum_ml_classification.ipynb` — full, pre-executed notebook (code + outputs + plots + discussion). Ready to open, re-run, or upload as-is.

## How to run (Kaggle / any Jupyter environment)
1. Upload the `.ipynb` to Kaggle (or open locally with Jupyter).
2. Make sure internet is **ON** in the notebook's settings (needed for the `pip install` cell — Qiskit is not preinstalled on Kaggle).
3. Run all cells top to bottom. Total runtime ≈ 2–3 minutes (the QSVM and QNN cells are the slow ones, ~20–70s each — this is expected and discussed in the notebook).

## What's inside
1. **Setup & data loading** — Breast Cancer Wisconsin dataset (569 samples, 30 features).
2. **Preprocessing** — `StandardScaler` → PCA (30 → 3 features) → scaled to `[-π/2, π/2]` for quantum angle encoding. The scaling-range choice is empirically justified in-notebook.
3. **Classical baselines** — SVM (RBF) and Logistic Regression, on both the full dataset and a subsample matched to the quantum models' training size (for a fair comparison).
4. **QSVM** — `ZZFeatureMap` + `FidelityQuantumKernel` + `QSVC` (Qiskit's drop-in quantum-kernel replacement for `sklearn.svm.SVC`). Includes a kernel-matrix heatmap.
5. **QNN** — `RealAmplitudes` variational ansatz + `EstimatorQNN` + `NeuralNetworkClassifier`, trained with `COBYLA` from 5 random restarts (best selected by *training* accuracy only, to avoid test-set leakage). Includes a convergence plot.
6. **Results comparison** — accuracy/precision/recall/F1/runtime table + bar charts.
7. **2D decision boundary visualization** — classical SVM vs QSVM, side by side.
8. **Confusion matrices** for all three models on the same test subsample.
9. **Limitations & honest discussion** — explicit, not hidden: no quantum advantage claimed, small training set, simulator-only, scaling/depth sensitivity, QNN optimization instability, qubit budget.

## Headline results (this run; simulator, `random_state=42`)

| Model | Accuracy | Train/predict time |
|---|---|---|
| Classical SVM (RBF, full train set) | ~0.94 | <0.01s |
| Logistic Regression (full train set) | ~0.96 | <0.01s |
| Classical SVM (RBF, subsampled — fair vs. quantum) | ~0.88 | <0.01s |
| **QSVM** | **~0.88** (matches subsampled classical) | ~18s |
| **QNN** (best of 5 restarts) | **~0.70** | ~67s |

Exact numbers can shift slightly on re-run (randomized subsampling / optimizer restarts) but the qualitative story is stable: **QSVM is competitive with classical on a matched small-sample regime; QNN is noticeably harder to train; both quantum methods are far slower than classical, with no accuracy advantage over the full-data classical models.**

## Environment (pinned versions, validated)
```
qiskit==2.5.2
qiskit-machine-learning==0.9.1
qiskit-aer
scikit-learn
matplotlib
pandas
```

## Limitations (see notebook Section 11 for full discussion)
- Simulator-only — no real IBM Quantum hardware was used.
- Quantum models trained on a 100-sample subsample (not the full 455-sample training set) due to O(n²) quantum-kernel cost.
- Reduced to 3 features via PCA to keep the qubit count tractable — discards some of the original signal.
- No claim of "quantum advantage" — this is a skills-demonstration project showing classical AI/ML fused with quantum computing concepts, not a production recommendation.

## Possible extensions
- Run on real IBM Quantum hardware via `qiskit-ibm-runtime`, compare simulator vs. real-device noise.
- Try `PegasosQSVC` for faster QSVM training on larger samples.
- Systematic sweep of entanglement structure / circuit depth with cross-validation.

# Five-model comparison: fixed 200k experiment

All five rows use the deterministic 200,000-row training subset, 20,000-row
validation subset, and the same 178,083-row test set. The primary threshold is
0.5 and the random seed is 42. Transformer results were read from the completed
result archives; traditional-model results use the new 2026-08-16 CPU rerun.

The legacy archive names `distilbert_full.zip`, `bert_base_full.zip`, and
`hatebert_full.zip` are misleading: their configs are the 200k/20k/178083
runs shown here. The genuine full-data sensitivity results are kept separately
under `results/full_data/`.

| Model | Family | Accuracy | Toxic precision | Toxic recall | Toxic F1 | Macro F1 | TN / FP / FN / TP | Training time (s) |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Logistic Regression | Traditional ML | 0.9060 | 0.4469 | 0.7378 | 0.5567 | 0.7520 | 150,832 / 13,006 / 3,735 / 10,510 | 5.62 |
| Linear SVM | Traditional ML | 0.9106 | 0.4590 | 0.6595 | 0.5413 | 0.7459 | 152,765 / 11,073 / 4,851 / 9,394 | 10.61 |
| DistilBERT | General-purpose Transformer | 0.9500 | 0.7144 | 0.6239 | 0.6661 | 0.8195 | 160,286 / 3,552 / 5,358 / 8,887 | 3,126.09 |
| BERT-base | General-purpose Transformer | 0.9501 | 0.7050 | 0.6474 | 0.6750 | 0.8240 | 159,979 / 3,859 / 5,023 / 9,222 | 6,006.60 |
| HateBERT | Task-specific Transformer | 0.9484 | 0.6781 | 0.6761 | 0.6771 | 0.8245 | 159,266 / 4,572 / 4,614 / 9,631 | 6,071.68 |

The machine-readable confusion-matrix artifacts are in
`results/fixed_200k/comparison/confusion_matrices/`, with one JSON and PNG
per model. Matrix order is rows=true labels and columns=predicted labels.

For the new traditional-model rerun, the full pipeline times (shared TF-IDF,
model fitting, and prediction) are 44.42 s for Logistic Regression and 49.42 s
for Linear SVM. The main table retains model-fitting time so its timing column
matches the Transformer training-time definition as closely as possible.

## Initial interpretation

- Transformer models substantially improve accuracy and macro F1 over the two
  TF-IDF baselines.
- HateBERT has the highest toxic-class F1 and toxic recall among the three
  Transformer models, while BERT-base has slightly higher accuracy.
- The traditional models are much faster to train, but have lower toxic-class
  precision and F1.
- DistilBERT is close to BERT-base and HateBERT while requiring roughly half
  the recorded training time of BERT-base/HateBERT in these runs.

These are descriptive comparisons. Because the runs were produced on different
hardware environments (CPU for traditional ML and 2x Tesla T4 for
Transformers), training time should not be treated as a hardware-controlled
benchmark.

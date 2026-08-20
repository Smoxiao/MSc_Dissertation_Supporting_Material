# Full-data five-model results

This is the secondary full-data sensitivity comparison. It does not replace the
primary fixed-200k comparison in `results/fixed_200k/comparison/`.

## Shared data contract

All five models use the same cleaned Civil Comments split and label definition
`label = 1 when target >= 0.5`.

| Split | Rows |
|---|---:|
| Train | 1,424,657 |
| Validation | 178,082 |
| Test | 178,083 |

The BERT-base and HateBERT prediction exports contain 178,083 unique test IDs
in exactly the same order as the fixed test file, with no ID, label, or text
mismatches. Their confusion matrices and metrics were independently recomputed
from the prediction files.

## Results

The confusion-matrix columns are ordered as `TN / FP / FN / TP`.

| Model | Accuracy | Toxic precision | Toxic recall | Toxic F1 | Macro F1 | TN / FP / FN / TP | Recorded training time |
|---|---:|---:|---:|---:|---:|---|---:|
| Logistic Regression | 0.9000 | 0.4333 | **0.8110** | 0.5648 | 0.7542 | 148,727 / 15,111 / 2,692 / 11,553 | 63.78 s |
| Linear SVM | 0.8950 | 0.4176 | 0.7928 | 0.5470 | 0.7438 | 148,085 / 15,753 / 2,951 / 11,294 | 122.41 s |
| DistilBERT | 0.9536 | 0.7400 | 0.6477 | 0.6908 | 0.8329 | 160,596 / 3,242 / 5,018 / 9,227 | 10,529.67 s |
| BERT-base | **0.9547** | **0.7607** | 0.6328 | 0.6909 | 0.8332 | 161,002 / 2,836 / 5,231 / 9,014 | 21,724.54 s |
| HateBERT | 0.9540 | 0.7440 | 0.6476 | **0.6925** | **0.8338** | 160,664 / 3,174 / 5,020 / 9,225 | 21,483.10 s |

**DistilBERT in this upper table is the new Kaggle rerun from
`distilbert_full_controlled.zip`: 1,424,657 training rows, 1 epoch, toxic F1
0.6908. The earlier full-data result is not used in this table.**

HateBERT has the highest toxic-class F1 and macro F1. BERT-base has the highest
accuracy and toxic precision. DistilBERT remains close to both larger models.
The traditional models have higher toxic recall but substantially lower toxic
precision and toxic F1.

## Fixed-200k comparison (same test set)

This table is placed directly below the full-data results for reference. These
runs used 200,000 training rows and 20,000 validation rows; they must not be
treated as additional full-data runs. The test set is the same 178,083 rows.
The DistilBERT value `0.6661` below is the original fixed-200k result, not the
new full-data rerun. The traditional-model rows are the new 2026-08-16 200k
CPU rerun.

| Model | Accuracy | Toxic precision | Toxic recall | Toxic F1 | Macro F1 | TN / FP / FN / TP | Recorded training time |
|---|---:|---:|---:|---:|---:|---|---:|
| Logistic Regression | 0.9060 | 0.4469 | 0.7378 | 0.5567 | 0.7520 | 150,832 / 13,006 / 3,735 / 10,510 | 5.62 s |
| Linear SVM | 0.9106 | 0.4590 | 0.6595 | 0.5413 | 0.7459 | 152,765 / 11,073 / 4,851 / 9,394 | 10.61 s |
| DistilBERT | 0.9500 | 0.7144 | 0.6239 | 0.6661 | 0.8195 | 160,286 / 3,552 / 5,358 / 8,887 | 3,126.09 s |
| BERT-base | 0.9501 | 0.7050 | 0.6474 | 0.6750 | 0.8240 | 159,979 / 3,859 / 5,023 / 9,222 | 6,006.60 s |
| HateBERT | 0.9484 | 0.6781 | 0.6761 | 0.6771 | 0.8245 | 159,266 / 4,572 / 4,614 / 9,631 | 6,071.68 s |

The machine-readable version containing both experiment sizes is
`model_comparison_combined.csv`.

## Change from the fixed-200k experiment

| Model | 200k toxic F1 | Full-data toxic F1 | Absolute change |
|---|---:|---:|---:|
| Logistic Regression | 0.5567 | 0.5648 | +0.0082 |
| Linear SVM | 0.5413 | 0.5470 | +0.0058 |
| DistilBERT | 0.6661 | 0.6908 | +0.0247 |
| BERT-base | 0.6750 | 0.6909 | +0.0159 |
| HateBERT | 0.6771 | 0.6925 | +0.0154 |

All five models improve in toxic F1 with the larger training set. The largest
gain is observed for DistilBERT.

## Configuration and interpretation limits

- The full-data inputs and seed are shared, but compute settings are not fully
  controlled: the current DistilBERT run and BERT-base/HateBERT ran for 1 epoch
  with 2 Tesla T4 GPUs and a per-device batch size of 32. The earlier
  full-data result is not used in this table.
- The Transformer runs used maximum length 128, learning rate `2e-5`,
  weight decay `0.01`, warmup ratio `0.1`, FP16, and validation toxic F1 for
  checkpoint selection. DistilBERT selected checkpoint 22261; BERT-base and
  HateBERT selected checkpoint 44521.
- Traditional-model times are model-fit time and exclude their shared 198.60 s
  TF-IDF vectorisation. Transformer times are recorded trainer training time.
  Runtime values are therefore descriptive, not a controlled benchmark.
- The formal DistilBERT result folder contains predictions, error files, reports,
  metrics, runtime information, and plots. The separate archive retains the
  best model but excludes the large trainer checkpoints.
- Because epoch count and GPU setup differ, this table should be described as a
  full-data sensitivity result set, not as a stricter replacement for the
  controlled 200k comparison.

## Result locations

- Logistic Regression: `results/full_data/logistic_regression/`
- Linear SVM: `results/full_data/linear_svm/`
- DistilBERT: `results/full_data/distilbert/`
- BERT-base: `results/full_data/bert_base/`
- HateBERT: `results/full_data/hatebert/`
- Archive names and provenance are recorded in the surrounding result folders.

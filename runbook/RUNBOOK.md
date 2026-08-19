# Dissertation Supporting Material Runbook

## Data preparation

Open notebook 01 from the supporting-material root. It creates or checks:

- the cleaned data
- the stratified full train, validation and test split
- the fixed 200k train and 20k validation subset
- the shared 178,083-row test set

Existing files are retained by the notebook.

## Model experiments

Notebook 02 contains the fixed-200k Logistic Regression and Linear SVM  
implementation. Notebook 06 contains the full-data version.

Notebooks 03 to 05 contain the fixed-200k Transformer implementations.  
Notebooks 07 to 09 contain the full-data Transformer implementations.

Each Transformer notebook is one model and one data regime. Full-data  
Transformer training is expensive, so run models individually, not with Run  
All. Approximate historical training times are written at the top of each  
Transformer notebook.

Kaggle-ready Transformer notebooks are kept under  
`runbook/kaggle_notebooks/`. They follow the original Kaggle workflow: attach  
the prepared split dataset, enable a GPU, run one model notebook, and download  
the results ZIP from `/kaggle/working/`.

The 200k notebooks read `experiment_train_200k.csv`,  
`experiment_validation_20k.csv` and `test.csv` from the `test-20k` Kaggle  
dataset. The full-data notebooks read `train.csv`, `validation.csv` and  
`test.csv` from the `full-data` Kaggle dataset. Each notebook has a fixed model  
and output folder; no model selector is required.

## Evaluation and error analysis

Notebook 10 recalculates metrics and confusion matrices from the final  
prediction files. It does not load a saved comparison table just to display it.

Notebook 11 verifies the full-data error pool of 24,154, the fixed-200k  
comparison pool of 23,795, the existing general 100 sample and the existing  
targeted 200 sample. It does not resample reviewed cases.

Completed outputs are under results/. Local reruns are written separately  
under reproduced_runs/.

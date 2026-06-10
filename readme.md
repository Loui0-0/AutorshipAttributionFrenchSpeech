# README: SpeakPred



## Installation

### Dependencies
Install required packages:
```pip install numpy scikit-learn pandas torch sentence-transformers gensim datasets matplotlib```

### Word2Vec model (optional)
For the "word2vec" feature extraction method a gensim keyed vector in the binary format is requiered

## Dataset Format

The script expects a csv file with at least two columns:

| Column name | Description |
| :--- | :--- |
| **text** | The document text (string) |
| **label** | The author/speaker identifier (string or integer) |

---

## Usage

**Bow**
```python speakpred.py --dataset_path datasets/ath_attrb_10.csv --stop_words_path "datasets/fr.txt" --feature_extraction_method bow --classifier log_reg --pp clf__C=10```

**Use Word2Vec embeddings**
```py speakpred.py --dataset_path datasets/ath_attrb_10.csv --word2vec_path="models\frWac_non_lem_no_postag_no_phrase_200_cbow_cut0.bin" --feature_extraction_method="w2v" --classifier="ridge" --save_dir="logs" --pp clf__alpha=0.01 --pp w2v__pooling_method="concat"```

**Perform grid search on classifier only**
``` py speakpred.py --dataset_path datasets/ath_attrb_10.csv --stop_words_path "datasets/fr.txt" --feature_extraction_method w2v --classifier svc_linear --pp clf__C=10 --gridsearch ```

**Run cross-validation (5 folds)**
```python speakpred.py --dataset_path datasets/ath_attrb_10.csv --feature_extraction_method bow --classifier log_reg --crossvalscore --crossvalscore_cv 5```

**Override a pipeline parameter (e.g., KNN neighbours)**
```python speakpred.py --dataset_path data.csv --feature_extraction_method bow --classifier knn --pp clf__n_neighbors=15```

---

## Command-line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| --dataset_path | str | speakpred_dataset.csv | Path to CSV dataset |
| --word2vec_path | str | "" | Path to Word2Vec binary model (.bin) |
| --stop_words_path | str | models/fr.txt | Path to stop words file (one word per line) |
| --feature_extraction_method | str | bow | One of: bow, w2v, word2vec, staticembedding, transformer, bert |
| --classifier | str | log_reg | See Supported Classifiers below |
| --save_dir | str | logs | Directory to save outputs (logs, models, plots) |
| --gridsearch | flag | False | Perform grid search on pipeline parameters |
| --gridsearch_full_pipeline | flag | False | Search over both feature extractor and classifier (slow) |
| --gridsearch_n_jobs | int | 1 | Number of parallel jobs for grid search |
| --crossvalscore | flag | False | Perform k-fold cross-validation |
| --crossvalscore_cv | int | 5 | Number of folds for cross-validation |
| --plot_feature_effects | flag | True | Generate feature effect plot (BoW/Linear only) |
| --pipeline_parameters / --pp | key=val | {} | Set any sklearn pipeline parameter |

---

## Supported Classifiers

| Argument name | Sklearn class |
| :--- | :--- |
| multinomialnb | MultinomialNB |
| lda | LinearDiscriminantAnalysis |
| log_reg | LogisticRegression |
| knn | KNeighborsClassifier |
| svc_linear | LinearSVC |
| ridge | RidgeClassifier |
| decision_tree | DecisionTreeClassifier |

---

## Outputs

All results are stored inside ```save_dir``` (default ```logs/```) in a timestamped subfolder:
```
logs/
└── 2025-01-27_15-30-00/
    ├── config.json                  # Full configuration used
    ├── logs.json                    # Final test score and benchmark report
    ├── top_features_per_authors.csv # If requested 
    └── (gridsearch subfolder)/
        ├── grid_results.csv
        ├── best_params.json
        ├── report.txt
        └── config.json

```

---

## Important Notes

* **RAM usage:** don't use ```--gridsearch_full_pipeline``` for transformer/Word2Vec to avoid instantiating a model for every worker.

* **BERT model** if you do not have a cuda ready GPU do not attempt, it will never compute.

---


## Author
Louis de Domingo - 2026 -

## Context
Machine Learning Project Course at Université Paris Dauphine-PSL in the second smester of M1, IDD.
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from speakpred import SpeakPred, DatasetHandler 

"""
Program we used to generate the result table for the BERT/Transformer section.
we compute the embeddings once using pooling="none" and perform the pooling externally
to avoid recomputing. 
"""

DATASET_CONFIG = {
    "dataset_path": "ath_attrb_40.csv",
}

CV_CONFIG = {
    "cv": 5,
    "n_jobs": [1, 1, 1, 1] 
}

PIPELINE_PARAMS = {
    "log_reg": {'clf__C': .1, 'clf__penalty': 'l1', 'clf__solver': 'saga'},
    "knn" :{'clf__metric': 'cosine', 'clf__n_neighbors': 10, 'clf__weights': 'distance'},
    "decision_tree" :{'clf__criterion': 'gini', 'clf__max_depth': 10, 'clf__min_samples_split': 10},
    "svc_linear":{'clf__C': 0.1, 'clf__dual': False, 'clf__loss': 'squared_hinge'}
}

FEATURE_CONFIGS = {
    "Mean": "mean",
    "Max": "max",
    "Sum": "sum",
    "Concat": "concat",
}

CLASSIFIERS = [
    # "log_reg",
    "knn",
    "decision_tree",
    "svc_linear",
]


data_handler = DatasetHandler(**DATASET_CONFIG)
X_train, X_test, y_train, y_test = data_handler.get_train_test_split()

sp = SpeakPred(
    feature_extraction_method="bert", 
    pipeline_parameters={
        "bert__pooling_method":"none", # tells it to not pool the vectors
        "bert__verbose": True 
    }
)
sp.set_model()

print("Extracting BERT chunk embeddings, .")
X_chunks = sp.model[:-1].fit_transform(X_train)

print("chunk extraction completed")

print("Precomputing each pooling method...")
transformed_data = {}

for feat_name, method in FEATURE_CONFIGS.items():
    pooled_X = []
    
    for chunk_emb in X_chunks:
        # Safety check in case of empty embeddings
        if chunk_emb.shape[0] == 0:
            v_size = sp.trans_model.get_sentence_embedding_dimension()
            size = v_size * 3 if method == "concat" else v_size
            pooled_X.append(np.zeros(size))
            continue
            
        if method == "mean":
            pooled = chunk_emb.mean(axis=0)

        elif method == "max":
            pooled = chunk_emb.max(axis=0)

        elif method == "sum":
            pooled = chunk_emb.sum(axis=0)

        elif method == "concat":
            pooled = np.concatenate([
                chunk_emb.mean(axis=0),
                chunk_emb.max(axis=0),
                chunk_emb.sum(axis=0)
            ])
            
        pooled_X.append(pooled)
        
    transformed_data[feat_name] = np.array(pooled_X)



results = []

for idx, (feat_name, method) in enumerate(FEATURE_CONFIGS.items()):
    
    X_tr = transformed_data[feat_name]
    
    
    n_jobs_current = CV_CONFIG["n_jobs"][idx]

    for clf_name in CLASSIFIERS:
        print(f"running: BERT_{feat_name} + {clf_name}")

        clf_base = clone(SpeakPred.DCT_CLF_METHODS[clf_name])
        clf_pipeline = Pipeline([('clf', clf_base)])

        params = PIPELINE_PARAMS.get(clf_name, {})
        
        # load correct parameters
        clf_pipeline.set_params(**params)
        
        cv_results = cross_validate(
            clf_pipeline,
            X_tr,
            y_train,
            cv=CV_CONFIG["cv"],
            n_jobs=n_jobs_current,
            scoring={'accuracy': 'accuracy', 'f1_macro': 'f1_macro'}
        )

        acc = np.mean(cv_results["test_accuracy"])
        f1 = np.mean(cv_results["test_f1_macro"])

        results.append({
            "Feature": f"BERT_{feat_name}",
            "Classifier": clf_name,
            "Accuracy": acc,
            "F1": f1
        })
        
        print(results[-1])

df = pd.DataFrame(results)
print("\n FULL RESULTS:")
print(df)
df.to_csv("BERTresults40.csv", index=False)
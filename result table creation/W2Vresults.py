import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from speakpred import SpeakPred, DatasetHandler 

"""
Program we used to generate the result table in the Word2vec section
"""


DATASET_CONFIG = {
    "dataset_path": "ath_attrb_40.csv",
    "stop_words_path": "datasets/fr.txt",
}

CV_CONFIG = {
    "cv": 5,
    "n_jobs": [-1, -1, -1, 12]
}

PIPELINE_PARAMS = {
    "log_reg": {
        "clf__C": 10,
        "clf__penalty": "l2",
    },

    "knn" :{
        "clf__metric": "cosine",
        "clf__n_neighbors": 5
    },

    "decision_tree" :{
        "clf__max_depth": 10
    },

    "svc_linear":{
        "clf__C": 0.001,
        "clf__dual": False,
    },
}

FEATURE_CONFIGS = {
    "Mean":"mean",
    "Max": "max",
    "Sum":"sum",
    "Concat":"concat",
}

CLASSIFIERS = [
    "log_reg",
    "knn",
    "decision_tree",
    "svc_linear",
]

data_handler = DatasetHandler(**DATASET_CONFIG)
X_train, X_test, y_train, y_test = data_handler.get_train_test_split()

# instantiate the keyed vector only once 
sp = SpeakPred(
    feature_extraction_method="w2v",
    word2vec_path="models/frWac_non_lem_no_postag_no_phrase_200_cbow_cut0.bin",
    stopwords=data_handler.stop_word_set
)
sp.set_model()

# we perform the feature extraction only once
transformed_data = {}
for feat_name, method in FEATURE_CONFIGS.items():
    sp.model.set_params(w2v__pooling_method=method)
    transformed_data[feat_name] = sp.model[:-1].fit_transform(X_train)


results = []

for idx, (feat_name, method) in enumerate(FEATURE_CONFIGS.items()):
    
    X_tr = transformed_data[feat_name]
    n_jobs_current = CV_CONFIG["n_jobs"][idx]

    for clf_name in CLASSIFIERS:
        print(f"running: {feat_name} + {clf_name}")

        clf_base = clone(SpeakPred.DCT_CLF_METHODS[clf_name])
        clf_pipeline = Pipeline([('clf', clf_base)])

        params = PIPELINE_PARAMS.get(clf_name, {})
        
        # load the correct parameter
        clf_pipeline.set_params(**params)
        
        # here we compute the cross val score outside the SpeakPred class for simplicity as we are not following the tandart use case for speed 
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
            "Feature": feat_name,
            "Classifier": clf_name,
            "Accuracy": acc,
            "F1": f1
        })
        
        print(f"{feat_name} + {clf_name}", results[-1])

df = pd.DataFrame(results)
print("\n FULL RESULTS:")
print(df)
df.to_csv("W2Vresult40.csv", index=False)
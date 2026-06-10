import numpy as np
import pandas as pd

from speakpred import SpeakPred, DatasetHandler 


"""
program we used to compute the result table from the BOW section 
"""


DATASET_CONFIG = {
    "dataset_path": "ath_attrb_10.csv",
    "stop_words_path": "datasets/fr.txt",
}

CV_CONFIG = {
    "cv": 5,
    "n_jobs": [5, 2, 1] # each n-gram parameter use exponentialy more ram so we scale down the number of workers for higher n
}

PIPELINE_PARAMS = {
    "log_reg": {
        "bow__use_idf": False,
        "bow__binary":  False,
        "clf__C": 10,
        "clf__penalty": "l1"
    },

    "multinomialnb": {
        "bow__use_idf": True,
        "bow__binary":  True,
        "clf__alpha": 0.001
    },

    "knn" :{
        "bow__use_idf": False,
        "bow__binary":  True,
        "clf__metric": "cosine",
        "clf__n_neighbors": 10
    },

    "decision_tree" :{
        "bow__use_idf": False,
        "bow__binary":  False,
    },

    "svc_linear":{
        "bow__use_idf": False,
        "bow__binary":  True,
        "clf__C": 100,
        "clf__dual": False,
    },

}



FEATURE_CONFIGS = {
    "BOW":      (1, 1),
    "BOW2gram": (1, 2),
    "BOW3gram": (1, 3),
}

CLASSIFIERS = [
    "log_reg",
    "multinomialnb",
    "knn",
    "decision_tree",
    "svc_linear",
]

data_handler = DatasetHandler(**DATASET_CONFIG)
X_train, X_test, y_train, y_test = data_handler.get_train_test_split()

results = []

for idx, (feat_name, ngram) in enumerate(FEATURE_CONFIGS.items()):
    for clf in CLASSIFIERS:

        print(f"running: {feat_name} + {clf}")

        params = {
            "bow__ngram_range": ngram
        }

        # setting the correct parameters
        params.update(PIPELINE_PARAMS.get(clf, {}))

        sp = SpeakPred(
            classifier=clf,
            feature_extraction_method="bow",
            pipeline_parameters=params,
            stopwords=data_handler.stop_word_set
        )

        print(sp.model,"\n",  params)

        sp.set_model()

        cv_results, _ = sp.test_model_via_crossvalscore(
            X_train,
            y_train,
            cv=CV_CONFIG["cv"],
            n_jobs=CV_CONFIG["n_jobs"][idx]
        )

        acc = np.mean(cv_results["test_accuracy"])
        f1 = np.mean(cv_results["test_f1_macro"])

        results.append({
            "Feature": feat_name,
            "Classifier": clf,
            "Accuracy": acc,
            "F1": f1
        })
        print(f"{feat_name} + {clf}", results[-1])


# results
df = pd.DataFrame(results)
print("\n FULL RESULTS:")
print(df)
df.to_csv("BOWresults10.csv", index=False)
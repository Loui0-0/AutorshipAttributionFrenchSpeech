import argparse
from datetime import datetime
import os
import pickle
import json
import sys
import numpy as np
import torch
import logging
from collections import Counter
import pandas as pd
from time import gmtime, strftime

import ast
from sklearn.model_selection import cross_validate

from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from sklearn.tree import DecisionTreeClassifier

import logging
from datasets import load_dataset
from sklearn.base import clone

# setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# for parsing the pipeline arg passed in the command line (like --pp clf_alpha=0.1)
class ConvertParametersAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):

        # values is the string "key=value"
        try:
            key, val_str = values.split('=', 1)
        except ValueError:
            raise argparse.ArgumentError(self, "format must be key=value")

        # Try to convert the value to a Python literal
        try:
            val = ast.literal_eval(val_str)
        except (ValueError, SyntaxError):

            # a ValueError would indicate that the name is not define meaning we are likely passing a method name like cosine
            val = val_str

        d = getattr(namespace, self.dest, {})
        d[key] = val
        setattr(namespace, self.dest, d)


def benchmark_classifier(model, X_test, y_test, labels_dict=None):
    y_pred = model.predict(X_test)

    # a dummy classifier will select at random classes according to their frequency this provides a baseline.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_test, y_test)
    baseline_acc = dummy.score(X_test, y_test)

    model_acc = accuracy_score(y_test, y_pred)
    model_f1 = f1_score(y_test, y_pred, average='macro')

    if labels_dict:
        display_names = [name for name, val in sorted(labels_dict.items(), key=lambda item: item[1])]
    else:
        display_names = None

    lines = []
    lines.append(f"full model desciption : {model}")
    lines.append(f"model accuracy:    {model_acc:.2%}")
    lines.append(f"random classifier: {baseline_acc:.2%}")
    lines.append(f"improvement:       {model_acc - baseline_acc:+.2%}")
    lines.append(f"macro F1-Score:    {model_f1:.4f}")
    lines.append("")
    lines.append("sklearn classification report:")
    lines.append(classification_report(y_test, y_pred, target_names=display_names))

    return "\n".join(lines)

def get_feature_effects(model, X_train_text, target_names=None, plot=False, n_feature_per_class=5):
    # extract the classifier
    clf = model.steps[-1][1]
    if not hasattr(clf, 'coef_'):
        raise ValueError("the classifier must have a coef_ attribute.")

    # extract feature names (only works with bag of word)
    feature_names = None
    for name, step in model.named_steps.items():
        if hasattr(step, 'get_feature_names_out'):
            feature_names = step.get_feature_names_out()
            break
    
    if feature_names is None:
        raise ValueError("could not find a vectorizer step with feature names.")


    # transform text to get frequencies
    transformer = Pipeline(model.steps[:-1])
    X_train_tr = transformer.transform(X_train_text)

    # average feature effects: coef. * mean freq.
    avg_effects = clf.coef_ * np.asarray(X_train_tr.mean(axis=0)).ravel()

    classes = clf.classes_
    class_labels = list(target_names) if target_names is not None else [str(cls) for cls in classes]
    
    top = pd.DataFrame()
    top_indices = np.array([], dtype=int)

    for i, label in enumerate(class_labels):
        top_idx = np.argsort(avg_effects[i])[-n_feature_per_class:][::-1]
        
        top[f"{label}"] = feature_names[top_idx]
        top[f"{label}_val"] = avg_effects[i, top_idx]
        
        top_indices = np.unique(np.concatenate((top_indices, top_idx)))

    predictive_words = feature_names[top_indices]


    return None, top

# handles dataset, uses the dataset class to make it compatible with the hugging face ecosystem
# when loading immediatly performs the train test split
class DatasetHandler:
    def __init__(self,
                 dataset_path: str = "",
                 stop_words_path: str = "",
                 label_col_name: str = "label",
                 feats_col_name: str = "text",
                 **kwargs
                 ):
        
        self.stop_words_path = stop_words_path
        self.dataset_path = dataset_path
        self.label_col_name = label_col_name
        self.feats_col_name = feats_col_name 

        self.dataset = None
        self.stop_word_set = set()
        

        self.dct_label_to_id = {}
        self.dct_id_to_label = {}
        
        self.load_stopwords()
        self.load_dataset()

    def load_stopwords(self):
        if not self.stop_words_path:
            return
        
        logging.info("loading stop words...")
        try:
            with open(self.stop_words_path, 'r', encoding='utf8') as f:
                self.stop_word_set = set(line.rstrip() for line in f)
            logging.info("Success !")
        except Exception as e:
            logging.error(e)

    def load_dataset(self):
        logging.info("loading dataset...")
        try:

            raw_ds = load_dataset("csv", data_files=self.dataset_path)["train"]
            
            # build label maps
            unique_labels = raw_ds.unique(self.label_col_name)
            self.dct_label_to_id = {label: i for i, label in enumerate(unique_labels)}
            self.dct_id_to_label = {i: label for i, label in enumerate(unique_labels)}
            
            logging.info(f"label mapping created: {self.dct_label_to_id}")

            # lazy split on load 
            self.dataset = raw_ds.train_test_split(test_size=0.2, seed=67)
            
        except Exception as e:
            logging.fatal(f"Failed to load dataset: {e}")
            exit(0)
        
        logging.info("loading, mapping, and splitting dataset... Done")

    def get_encoded_dataset(self):
            logging.info("encoding labels to ids...")

            def map_labels(example):
                example[self.label_col_name] = self.dct_label_to_id[example[self.label_col_name]]
                return example

            encoded_ds = self.dataset.map(map_labels)
            
            logging.info("encoding labels to ids... Done")
            return encoded_ds
    

    def get_train_test_split(self):
        """Returns (X_train, X_test, y_train, y_test)"""
        train_ds = self.dataset["train"]
        test_ds = self.dataset["test"]
        
        return ( 
            train_ds[self.feats_col_name], 
            test_ds[self.feats_col_name],
            train_ds[self.label_col_name], 
            test_ds[self.label_col_name]
        )
        
# class to handle easily tweak and test the different feature extraction / classifier combination
class SpeakPred:
    # if no path is provided will try to find these models in the current working directory  
    DEFAULT_MODELS = {"word2vec": "models/frWac_non_lem_no_postag_no_phrase_200_cbow_cut0.bin"}

    # here we define the available classifier with some base parameters
    DCT_CLF_METHODS = {
            "multinomialnb": MultinomialNB(
                alpha=0.01 
            ),

            "lda": LinearDiscriminantAnalysis(

            ),

            "log_reg": LogisticRegression(
                C=0.1,
                penalty="l1",

                max_iter=10000, 
                class_weight='balanced', 
                solver="saga", # saga best for large datasets
            ), 

            "knn": KNeighborsClassifier(
                n_neighbors=10, 
                metric='cosine'
            ),

            "svc_linear":  LinearSVC( class_weight='balanced', C = 100, dual=False), # should scale better to large numbers of samples than SVC(kernle='linear').

            "ridge": RidgeClassifier(alpha=.01, solver="sparse_cg"),   
                                        
            "decision_tree": DecisionTreeClassifier()

}
    
    # here we define for each classifier and feature extraction methods grids to search parameter in,
    # this is not the best implementation as currently th only way to change the parameters is to change the source code 
    # but it was good enought for our use
    DCT_GRID_PARAMS = {

        "MultinomialNB": {'clf__alpha': (1, 0.5, 0.1, 0.01, 0.001)},

        "LinearDiscriminantAnalysis":{
            'clf__alpha': (1, 0.5, 0.1, 0.01, 0.001)
        },

        "LogisticRegression": {
            'clf__penalty': ['l1', 'l2', 'elasticnet'], 
            'clf__C': (0.1, 1, 10, 100), 
            'clf__solver': ['saga'], 
            'clf__l1_ratio': [0.5]  
        },

        "KNeighborsClassifier": {
            'clf__n_neighbors': (3, 5, 10, 20),
            'clf__weights': ['uniform', 'distance'],
            'clf__metric': ['cosine', 'euclidean']
        },

        "LinearSVC": {
            'clf__C': (0.1, 1, 10, 100), 
            'clf__loss': ['squared_hinge'],
            'clf__dual': [False]
        },
        
        "TfidfVectorizer": {
            'bow__use_idf': (False, True),
            'bow__binary':  (True, False)
        },

        "RidgeClassifier": {
            'clf__alpha': (0.1, 1.0, 10.0)                  
        },

        "DecisionTreeClassifier": {
            'clf__criterion': ("gini", "entropy", "log_loss"),
            'clf__criterion': ("gini", "entropy"),
            'clf__max_depth': [None, 10, 20, 30],
            'clf__min_samples_split': [2, 5, 10]
        },


    }
    
    # for model saving reason we coudn't have anonimous functions
    tokenizer_ = TfidfVectorizer().build_analyzer()

    def __init__(self, 
                 classifier:str = 'log_reg',
                 feature_extraction_method:str = 'bow',
                 word2vec_path:str = "",
                 pipeline_parameters:dict = {},
                 stopwords: set = set(),
                 **kwargs
                 ):
        
        self.classifier = classifier
        self.feature_extraction_method = feature_extraction_method
        self.word2vec_path = word2vec_path
        self.pipeline_parameters = pipeline_parameters
        self.stopwords = stopwords
        
        # will contain a pipeline object
        self.model = None

        # external models
        self.w2v_model = None
        self.trans_model = None

    def load_external_models(self):
        fe_method = self.feature_extraction_method.lower()
        
        if fe_method in ["staticembedding", "word2vec", "w2v"]:
            self.w2v_model = self.load_KeyedVectorWord2vec()
            if self.w2v_model is None:
                logging.fatal("No word2vec model, cannot continue execution.")
                sys.exit()

        elif fe_method in ["transformer", "bert", "camembert"]:
            self.trans_model = SentenceTransformer("dangvantuan/sentence-camembert-base")
    

    def set_model(self):
        # this function builds the requested pipeline  

        clf_name = self.classifier

        if clf_name not in SpeakPred.DCT_CLF_METHODS:
            logging.error(f"unsupported classifier: {clf_name}, using default: log_reg")
            classifier_template = SpeakPred.DCT_CLF_METHODS["log_reg"]
        else:
            classifier_template = SpeakPred.DCT_CLF_METHODS[clf_name]
        
        classifier = clone(classifier_template)

        # build requested pipeline
        fe_method = self.feature_extraction_method.lower()
        if fe_method in ["bow", 'bagofwords', 'bag of words']:
            self.model = SpeakPred.build_pipeline_bow(classifier, self.stopwords)
        
        elif fe_method in ["staticembedding", "word2vec", "w2v"]:
            
            
            if not self.w2v_model:
                self.load_external_models()

            self.model = SpeakPred.build_pipeline_staticembedding(self.w2v_model, classifier)
        
        
        elif fe_method in ["transformer", "bert", "camembert"]:
            if not self.trans_model:
                self.load_external_models()
            self.model = SpeakPred.build_pipeline_transformer(self.trans_model, classifier)


        if params := self.pipeline_parameters:
            self.model.set_params(**params)

    def train(self, X_train, y_train):
            logging.info("Starting training process...")
            
            # Calculate class distribution and support
            class_counts = Counter(y_train)
            num_classes = len(class_counts)
            total_samples = len(y_train)

            logging.info(f"Number of distinct classes: {num_classes}")
            logging.info(f"Total training samples: {total_samples}")
            logging.info(f"Class support:\n{class_counts}")
            

            self.model.fit(X_train, y_train)
            logging.info("Training done !")

    def score(self, X_test, y_test):
        if self.model == None:
            return -1
        return self.model.score(X_test, y_test)

    # this function use the grids define in the speakpred class and automaticaly buids 
    # the parameter grid for the current feat. extraction and clf. 
    # The classifier_only parameter allows to exclude the feature extraction from the gridsearch,
    # we did that because feature exctraction can be very long and when using external model
    # they have to be loaded in ram for each conccurent workers filling the ram instantly.
    def tune_model_via_gridsearchCV(self, X_train, y_train, cv=5, classifier_only=False,  n_jobs = -1):
        
        if self.model is None:
            self.set_model()
                
        model_to_tune = self.model

        if not model_to_tune.__sklearn_is_fitted__():
            
            logging.info("fitting the model prior to gridsearch...")
            model_to_tune.fit(X_train, y_train)
        
        # split the timeline into fe/clf and performs fe
        if classifier_only and isinstance(self.model, Pipeline):
            logging.info("grid search on classifier only, splitting the pipeline.")

            feature_extraction = Pipeline(self.model.steps[:-1])
            classifier = self.model.steps[-1][1]

            logging.info("extracting features.")
            X_tr = feature_extraction.fit_transform(X_train)

            model_to_tune = Pipeline([('clf', classifier)])
        else:
            X_tr = X_train
        
        
        # build grid parameters using SpeakPred.DCT_GRID_PARAMS
        grid_params = {}
        for _, step in model_to_tune.steps:

            step_name = type(step).__name__
            grid_params.update(SpeakPred.DCT_GRID_PARAMS.get(step_name, {}))


        logging.info(f"starting GridSearch with params: {grid_params} for {model_to_tune}")

        grid = GridSearchCV(
            model_to_tune,
            grid_params,
            cv=cv,
            n_jobs=n_jobs,
            verbose=100, # for easy debugging 
            return_train_score=True,
            scoring = "f1_macro"
        )

        grid.fit(X_tr, y_train)

        best_score = grid.best_score_
        best_estimator = grid.best_estimator_

        # build back the pipeline if needed
        if classifier_only:
            best_estimator = Pipeline(feature_extraction.steps.copy() + best_estimator.steps.copy())

        self.model = best_estimator

        logging.info(f"Best CV score: {best_score:.4f}")

        return grid_params, self.model, best_score, grid
    
    
    def test_model_via_crossvalscore(self, X_train, y_train,  cv=5, n_jobs=-1, scoring=None):

        if self.model is None:
            self.set_model()

        if scoring is None:
            scoring = {'accuracy': 'accuracy', 'f1_macro': 'f1_macro'}


        logging.info(f"Sarting cross validation data cv={cv}, n_jobs={n_jobs}, scoring={scoring}")
        cv_results = cross_validate(
            self.model, X_train, y_train,
            cv=cv, n_jobs=n_jobs,
            scoring=scoring,
            return_train_score=True,
            verbose=100
        )
        logging.info("cross validation done.")
    
        report_lines = []
        report_lines.append(f"Result of cross val (cv={cv}) for :\n{self.model}\n")
        report_lines.append("scores for test (mean +- std) :")
        for metric in scoring:
            scores = cv_results[f'test_{metric}']
            report_lines.append(f"  {metric}: {np.mean(scores):.4f} +- {np.std(scores):.4f}")
        report_lines.append("scores for train (mean +- std) :")
        for metric in scoring:
            scores = cv_results[f'train_{metric}']
            report_lines.append(f"  {metric}: {np.mean(scores):.4f} +- {np.std(scores):.4f}")
        report_lines.append("\n Test scores by fold :")
        
        n_folds = cv
        for fold in range(n_folds):
            fold_str = f"Fold {fold+1}: "
            fold_str += ", ".join(f"{metric}={cv_results[f'test_{metric}'][fold]:.4f}" for metric in scoring)
            report_lines.append(fold_str)

        return cv_results, report_lines

    #handles the w2v model loading
    def load_KeyedVectorWord2vec(self):
        modelfilepath = self.word2vec_path
        if modelfilepath == "":
            modelfilepath = SpeakPred.DEFAULT_MODELS["word2vec"]
            logging.info(f"no word2vec model provided using default : {modelfilepath}")
        
        logging.info("loading word2vec model...")
        try:
            model_w2v = KeyedVectors.load_word2vec_format(modelfilepath, binary=True, unicode_errors="ignore")
            logging.info("loading word2vec model... Done")
            return model_w2v
        except Exception as e:
            logging.fatal(f"could not load :{modelfilepath}. Error: {e}")
            return None

    
    def build_pipeline_bow(classifier: ClassifierMixin, stopwords: set = set()) -> Pipeline: 
        return Pipeline([
            ('bow', TfidfVectorizer(stop_words=list(stopwords))),
            ('clf', classifier)
        ])

    
    def build_pipeline_staticembedding(word_embbeding: KeyedVectors, classifier: ClassifierMixin) -> Pipeline:
        return Pipeline([
            ('tokenizer',FunctionTransformer(SpeakPred.texts_tokenizer, validate=False)),
            ('w2v', Word2vecSkTransformer(word2vec_model=word_embbeding)),
            ('clf', classifier)
        ])
    
    def build_pipeline_transformer(transformer_model: SentenceTransformer, classifier: ClassifierMixin) -> Pipeline:
            return Pipeline([
                ('bert', CustomTextEmbedderSkTransformer(model=transformer_model)),
                ('clf', classifier)
        ])
    
    def texts_tokenizer(texts):
        return [SpeakPred.tokenizer_(text) for text in texts]


   
# custom sklearn transformer for the w2v method
class Word2vecSkTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, word2vec_model: KeyedVectors, pooling_method: str = "mean"):
        self.word2vec_model = word2vec_model
        self.pooling_method = pooling_method 

    def fit(self, X, y=None):
        valid_methods = {"mean", "max", "sum", "concat"}
        if self.pooling_method not in valid_methods:
            raise ValueError(f"pooling_method must be in {valid_methods}")
        return self

    def transform(self, X):
        X_new = []
        v_size = self.word2vec_model.vector_size
        for tokenized_text in X:
            
            # embed each word
            embeddings = [self.word2vec_model[word] for word in tokenized_text if word in self.word2vec_model]
            
            # it is possible that no word had vectors available in that case we set it as 0
            if not embeddings:
                size = v_size * 3 if self.pooling_method == "concat" else v_size
                X_new.append(np.zeros(size))
                continue
            
            embeddings = np.array(embeddings)

            if self.pooling_method == "mean":
                embedding = embeddings.mean(axis=0)

            elif self.pooling_method == "max":
                embedding = embeddings.max(axis=0)

            elif self.pooling_method == "sum":
                embedding = embeddings.sum(axis=0)

            elif self.pooling_method == "concat":
                embedding = np.concatenate((embeddings.mean(axis=0), embeddings.max(axis=0), embeddings.sum(axis=0)))
            
       

            X_new.append(embedding)
        
        return np.array(X_new)

# on the same model as the w2v model
# works by embedding texts chunk by chunk and pooling the resulting embeddings
# making use of batches     
class CustomTextEmbedderSkTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model, pooling_method: str = "mean", section_overlap: int = 15, verbose: bool = False):
        self.model = model
        self.pooling_method = pooling_method
        self.section_overlap = section_overlap # number of token that overlap between consecutive chunks
        self.max_chunk_length = self.model.max_seq_length # here we set the max lenght to 512 the maximum context size for bert models
        self.verbose = verbose 

    def fit(self, X, y=None):
        valid_methods = {"mean", "max", "sum", "concat", "none"}
        if self.pooling_method not in valid_methods:
            raise ValueError(f"pooling_method must be in {valid_methods}")
        return self

    def transform(self, X):
        X_new = []
        device = self.model.device
        
        total_docs = len(X)
        
        for i, text in enumerate(X):
            
            # tokenize the texts and cut into chunks
            inputs = self.model.tokenizer(
                text,
                truncation=True,                  # ensure chunks larger than 512 are truncated
                max_length=self.max_chunk_length, # size of the chunks
                stride=self.section_overlap,      # size of the overlap
                return_overflowing_tokens=True,   # tell the tokenizer to proccess the text in chunk
                padding="max_length",             # chunks smaller than the 512 token limit will have padding tokens to be handled by the gpu
                return_tensors="pt"               # returns a pytorch tensor object to be handled by the gpu
            ).to(device) # sends value to the device that will perform the operations

         

            num_chunks = inputs["input_ids"].shape[0] 

            if self.verbose:
                print(f"[{i+1}/{total_docs}] processing speech... split into {num_chunks} chunks.", end='\r')

            # this is information we don't need
            inputs.pop("overflow_to_sample_mapping", None)

            with torch.no_grad(): # this tells the model to not keep track of information relevant only when training
                out = self.model(inputs) # computes the embeddings
                chunk_embeddings = out['sentence_embedding'].cpu().numpy() # retreive from the GPU the data

            if chunk_embeddings.shape[0] == 0:
                v_size = self.model.get_sentence_embedding_dimension()
                size = v_size * 3 if self.pooling_method == "concat" else v_size
                X_new.append(np.zeros(size))
                continue

            # now we can pool like in the w2v case
            if self.pooling_method == "mean":
                doc_emb = chunk_embeddings.mean(axis=0)

            elif self.pooling_method == "max":
                doc_emb = chunk_embeddings.max(axis=0)

            elif self.pooling_method == "sum":
                doc_emb = chunk_embeddings.sum(axis=0)

            elif self.pooling_method == "concat":
                doc_emb = np.concatenate([chunk_embeddings.mean(axis=0),
                                          chunk_embeddings.max(axis=0),
                                          chunk_embeddings.sum(axis=0)
                                        ])
            
            elif self.pooling_method == "none":

                doc_emb = chunk_embeddings
            
            X_new.append(doc_emb)

        return np.array(X_new) if self.pooling_method != "none" else X_new
 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog="25/26 ML Project - Authorship Attribution")

    parser.add_argument("--dataset_path",              type=str,  default="speakpred_dataset.csv",   help="path to the dataset file, must have text and label column name.")


    parser.add_argument("--word2vec_path",             type=str,  default="",                        help="path to the word2vec model file")
    parser.add_argument("--stop_words_path",           type=str,  default="models/fr.txt",           help="path to a stop word list")

    parser.add_argument("--feature_extraction_method", type=str,  default="bow",                     help="method used for feature extraction.")

    parser.add_argument("--classifier",                type=str,  default="log_reg",                 help="classification method.")
    
    parser.add_argument("--save_dir",                  type=str,  default="logs",                    help="where to save the model, the logs and the configuration")

    parser.add_argument("--gridsearch",                  action=argparse.BooleanOptionalAction,      help="performs a grid search on the parameter of the pipeline, saves the result in the save dir.")
    parser.add_argument("--gridsearch_full_pipeline",    action=argparse.BooleanOptionalAction,      help="When using the gridsearch embeddig the token for every fold is really expensive, this option allows to only tune the last step, the classifier.")
    parser.add_argument("--gridsearch_n_jobs",         type=int , default=1,                         help="Number of threads used by grid search, lower if ram is maxed out")
    
    parser.add_argument("--plot_feature_effects",        action=argparse.BooleanOptionalAction,      help="Plot feature effects for linear models (only bow)")
    
    parser.add_argument("--pipeline_parameters", "--pp", action=ConvertParametersAction, default={}, help="set a parameter for the pipeline using the sklearn pipeline parameter syntaxe , e.g. --pipeline_parameters clf__n_neighbors=3 or --pp ") 
    
    parser.add_argument("--crossvalscore",               action=argparse.BooleanOptionalAction,      help="performs cross validation")
    parser.add_argument("--crossvalscore_n_jobs",      type=int, default=1,                          help="Number of workers used by the cross valitdation, lower if ram is maxed out")
    parser.add_argument("--crossvalscore_cv",          type=int, default=5,                          help="Number of folds for the cross validation")

    args = parser.parse_args()
    config = vars(args)

    # load the dataset
    data_manager = DatasetHandler(**config)
    if config.get("only_build_dataset", False):
        exit(0)
    
    # retreive train test split
    X_train, X_test, y_train, y_test = data_manager.get_train_test_split()


    sp = SpeakPred(stopwords=data_manager.stop_word_set, **config)
    score = -1

    # logging logic from the provided template.
    # Logging to the disk only happens when calling the program from the command line
    # this ensure that the classes can be easily used externally without worrying about
    # clutter in the workspace    
    dir_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    out_dir = os.path.join(config["save_dir"], dir_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # it was a design choice to block normal training when gridsearch was requested. 
    if not config.get("gridsearch", False):
        sp.set_model()
        sp.train(X_train, y_train)
        score = sp.score(X_test, y_test)
    
    
    if config.get("gridsearch", False):
        grid_params, model, score, grid  = sp.tune_model_via_gridsearchCV(X_train, y_train, classifier_only = not config.get("gridsearch_full_pipeline", False), n_jobs = config.get("gridsearch_n_jobs", 1))
        print(model, score)

        # save results
        fe = sp.feature_extraction_method
        clf = sp.classifier
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dirname = f"{fe}_{clf}_{timestamp}_{score:.3f}"
        gsearch_out_dir = os.path.join(config["save_dir"], dirname)
        os.makedirs(gsearch_out_dir, exist_ok=True)

        logging.info(f"saving GridSearch logs to {gsearch_out_dir}")

        pd.DataFrame(grid.cv_results_).to_csv(
            os.path.join(gsearch_out_dir, "grid_results.csv"),
            index=False
        )

        with open(os.path.join(gsearch_out_dir, "best_params.json"), "w") as f:
            json.dump(grid.best_params_, f, indent=4)


        #! Commented because w2v models saves the full 2gb keyedvector
        # with open(os.path.join(gsearch_out_dir, "model.pkl"), "wb") as f:
        #     pickle.dump(self.model, f)

        with open(os.path.join(gsearch_out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        # add a little report to more easily read parameters
        with open(os.path.join(gsearch_out_dir, "report.txt"), "w") as f:
            f.write(f"grid search result on {grid_params}\n")
            f.write(f"best cv score: {score:.4f}\n\n")
            f.write("  best parameters:\n")
            for k, v in grid.best_params_.items():
                f.write(f"  {k}: {v}\n")

            f.write("benchmark on test data:\n")
            f.write(benchmark_classifier(sp.model, X_test, y_test))
    
    report = benchmark_classifier(sp.model, X_test, y_test)
    print(report)

    if config.get("crossvalscore", False):
        logging.info("Computing cross validation...")
        cv_results, report_lines = sp.test_model_via_crossvalscore(
            X_train, y_train,
            cv=config.get("crossvalscore_cv", 5),
            n_jobs=config.get("crossvalscore_n_jobs", 1)
        )

        fe = sp.feature_extraction_method
        clf = sp.classifier
        mean_acc = np.mean(cv_results['test_accuracy'])
        dirname = f"crossval_{mean_acc:.3f}"
        crossval_out_dir = os.path.join(out_dir, dirname)
        os.makedirs(crossval_out_dir, exist_ok=True)
        logging.info(f"s aving cross validation result in {crossval_out_dir}...")

        df_cv = pd.DataFrame(cv_results)
        df_cv.to_csv(os.path.join(crossval_out_dir, "cv_results.csv"), index=False)

        with open(os.path.join(crossval_out_dir, "report.txt"), "w", encoding='utf8') as f:
            f.write("\n".join(report_lines))

        with open(os.path.join(crossval_out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)


    if config.get("plot_feature_effects", True):
        if config["feature_extraction_method"].lower() in ["bow", "bagofwords", "bag of words"]:
            
            _, top = get_feature_effects(sp.model, X_train, n_feature_per_class=10000)


            plot_path = os.path.join(out_dir, "feature_effects.png")
            top_table_path = os.path.join(out_dir, "top_features_per_authors.csv")

            top.to_csv(top_table_path, index=False)

        else:
            logging.warning("feature effects only supported for bow model")

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f)
        
    # with open(os.path.join(out_dir, "model.pkl"), 'wb') as f:
    #     pickle.dump({"model": sp.model}, f)

    with open(os.path.join(out_dir, "logs.json"), "w") as f:
        json.dump({"score": score}, f)
        json.dump({"benchmark": report}, f)


    


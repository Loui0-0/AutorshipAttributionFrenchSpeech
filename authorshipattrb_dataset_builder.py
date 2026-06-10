from collections import Counter
import pandas as pd
import argparse
import os
import json
import sys
import logging
from datetime import datetime
DATASET_HEADER = {"label": "name", "feature": "text"}
LISTE_TYPE_DOCUMENT = ['Déclaration', 'Lettre']
LISTE_BAN_AUT = ["Présidence de la République", "Premier ministre", "Ministère des Affaires étrangères", "Ministère de l'Education nationale, de la Recherche et de la Technologie", "Ministère de l'Ecologie et de l'Environnement durable", "Secrétariat d'Etat aux Petites et Moyennes Entreprises, au Commerce, à l'Artisanat, aux Professions libérales et à la Consommation", "Secrétariat d'Etat aux Petites et Moyennes Entreprises, au Commerce, à l'Artisanat, aux Professions libérales et à la Consommation", "Ministère de l'Agriculture, de l'Alimentation, de la Pêche et des Affaires rurales", "Ministère de l'Éducation nationale", "Ministère de l'économie, des finances et de l'industrie", "Ministère de l'Equipement, des Transports, du Logement, du Tourisme et de la Mer", "Ministère de la Fonction publique, de la Réforme de l'Etat et de l'Aménagement du territoire", "Ministère de la Fonction publique et de la Réforme de l'Etat", "Ministère de l'Aménagement du territoire et de l'Environnement", "Ministère des Affaires étrangères et du Développement international", "Ministère de l'Intérieur, de la Sécurité intérieure et des Libertés locales", "Services du Premier ministre", "Secrétariat général du Gouvernement" , "Ministère des Affaires étrangères et européennes", "Ministère de l'Europe et des affaires étrangères"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog="25/26 ML Project - Vie Public Discours Analysis")

    parser.add_argument("--dataset_path", type=str,  default="datasets/viepubliquediscours.csv", help="path to the datset file, metadata + speeches.")
    parser.add_argument("--out_file",     type=str,  default="auth_attrb.csv",                   help="name of the outdataset")
    parser.add_argument("--n_author",     type=int,  default=10,                                 help="number of author (classes) extracted from the dataset. default is 10")
    


    args = parser.parse_args()
    config = vars(args)

    logging.info(f"Opening {config.get('dataset_path')} dataset...")
    df = pd.read_csv(config.get("dataset_path"), encoding='utf8')
    
    logging.info("building dataset...")
    df_solo = df[df["intervenants"].apply(lambda liste:len(eval(liste))) == 1].copy()
    df_solo["author"] = df_solo["intervenants"].apply(lambda liste_iter: eval(liste_iter)[0]["nom"])
    df_solo = df_solo[df_solo.type_document.isin(LISTE_TYPE_DOCUMENT)]

    df_speakpred = df_solo[["author", "speech"]].copy()
    df_speakpred.dropna(subset=["speech", "author"], inplace=True)

    entree_par_author_not_ban = Counter(df_speakpred[~df_speakpred.author.isin(LISTE_BAN_AUT)].author)
    author_sorted_par_n_entree = sorted(list(entree_par_author_not_ban.keys()), key=lambda aut: entree_par_author_not_ban[aut], reverse=True)
    top_author = author_sorted_par_n_entree[:config.get("n_author", 10)]

    df_speakpred = df_speakpred[df_speakpred.author.isin(top_author)]
    
    df_speakpred.rename(columns={
        "author" : "label", 
        "speech" : "text"
    }, inplace=True)

    logging.info(f"saving as {config.get('out_file')}")

    df_speakpred.to_csv(config.get("out_file"), index=False)
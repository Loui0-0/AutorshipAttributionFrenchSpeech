import json
import os
import requests
from bs4 import BeautifulSoup
import argparse
import time 


def getDiscours(url):

    r = requests.get(url)

    if not r.ok:
        return ""
    
    soup = BeautifulSoup(r.text, 'html.parser')
    # class containing the whole text
    discours_soup = soup.find('div', class_='field field--name-field-texte-integral field--type-text-long field--label-hidden field__item')
    if discours_soup is None:
        return ""
    return discours_soup.get_text(" ")


def read_public_discourse_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = "scraper for https://www.data.gouv.fr/datasets/metadonnees-des-discours-publics-de-vie-publique-fr")

    parser.add_argument("--discours_json_path", type = str, default = "",         help = "path to the discours metadatas")
    parser.add_argument("--download_folder",    type = str, default = "datasets", help = "path to the download folder")
    args = parser.parse_args()


    Discours_metadata = read_public_discourse_data(args.discours_json_path)
    n_total_discours = len(Discours_metadata)

    n_downloaded_discours = 0 
    sum_download_time = 0
    for discours in Discours_metadata:
        time_start = time.time()
        textefilepath = os.path.join(args.download_folder, str(discours["id"])) + ".txt"
        n_downloaded_discours += 1
        
        # allows to stop the download and start back later
        if os.path.exists(textefilepath):
            print(f"progresse: {round(n_downloaded_discours/n_total_discours * 100, 5)}%, averge download time : {round(sum_download_time/n_downloaded_discours, 2)} s, estimated remaining time: {round((sum_download_time/n_downloaded_discours)*(n_total_discours - n_downloaded_discours), 2)}")
           
            # if file exists check if it is empty before moving on
            with open(textefilepath, 'r', encoding='utf8') as d:
                if len(d.read())>0:
                    continue
        
        try:
            texte_discours = getDiscours(discours['url'])
            with open(textefilepath,  "w", encoding="utf8") as f:
                f.write(texte_discours)
        except:
            print("error for this file :" + discours["id"])
            
       

        time_end = time.time()

        sum_download_time += (time_end - time_start)
        
        print(f"progresse: {round(n_downloaded_discours/n_total_discours * 100, 5)}%, averge download time : {round(sum_download_time/n_downloaded_discours, 2)} s")


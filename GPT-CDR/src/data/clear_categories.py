
from pathlib import Path
import pandas as pd
import glob
import os

csv_folder = 'D:/CrossDomain_RecSys_LLM-main/data/processed/extra_cut'


        
def main():
    files = glob.glob(csv_folder + ".csv")
    for dirpath, dirnames, filenames in os.walk(csv_folder):
    # Usa un altro ciclo for per iterare su tutti i files nella sottocartella corrente
        for file in filenames:
            # Usa os.path.join per creare il percorso completo del file
            file_path = os.path.join(dirpath, file)
            # Aggiungi il file alla lista
            if ".csv" in file_path:
                files.append(file_path)
    print(files)
    
    for file in files:
        print("ok")
        df = pd.read_csv(file)
        #Per eliminare le pagine e amazon's
        df['brand'] = df['brand'].str.removesuffix(' Page')
        df['brand'] = df['brand'].str.replace("Visit Amazon's ", '')
        #per eliminare Books/movies/cds
        df['category'] = df['category'].str.replace("'Books', ", '')
        df['category'] = df['category'].str.replace("'Movies & TV', ", '')
        df['category'] = df['category'].str.replace("'CDs & Vinyl', ", '')
        df['rating'] = df['rating'].astype(int)
        df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
        
        df.to_csv(file, index=False)



if __name__ == '__main__':
    
    main()
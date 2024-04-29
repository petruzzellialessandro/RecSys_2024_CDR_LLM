import json
import gzip
import pandas as pd



BOOKS_PATH = "/home/rinaspata/data/amazon/amazon/ratings_Books.csv"
CDS_PATH = "/home/rinaspata/data/amazon/amazon/ratings_CDs_and_Vinyl.csv"
MOVIES_PATH = "/home/rinaspata/data/amazon/amazon/ratings_Movies_and_TV.csv"
BOOKS_METADATA_PATH = "/home/rinaspata/data/meta_Books.json.gz"
CDS_METADATA_PATH = "/home/rinaspata/data/meta_CDs_and_Vinyl.json.gz"
MOVIES_METADATA_PATH = "/home/rinaspata/data/meta_Movies_and_TV.json.gz"
CACHE_PATH = "/home/rinaspata/.cache"

COLUMN_HEADERS = ["user_id", "item_id", "rating", "timestamp"]


# -------------------
# Load datasets
# -------------------
ratings_books = pd.read_csv(BOOKS_PATH, names=COLUMN_HEADERS)
ratings_cds = pd.read_csv(CDS_PATH, names=COLUMN_HEADERS)
ratings_movies = pd.read_csv(MOVIES_PATH, names=COLUMN_HEADERS)


# -------------------
# Extract unique ids from datasets
# -------------------
books_ids = set(ratings_books['item_id'].unique())
cds_ids = set(ratings_cds['item_id'].unique())
movies_ids = set(ratings_movies['item_id'].unique())

print(f"Unique BOOKS ids: {len(books_ids)}")
print(f"Unique CDS ids: {len(cds_ids)}")
print(f"Unique MOVIES ids: {len(movies_ids)}")

# -------------------
# Load items metadata
# -------------------
print("\n\n-- Started to load BOOK metadata --")
data = []
i = 0
with gzip.open(BOOKS_METADATA_PATH, 'rt') as f:
    for l in f:
        json_obj = json.loads(l.strip())

        if (json_obj.get('asin') in books_ids) :
            data.append(json_obj) 
            i += 1
        
books_metadata = pd.DataFrame.from_dict(data)

print(f"Loaded BOOKS metadata: {i} / {len(books_ids)}")


print("\n\n-- Started to load CDS metadata --")
data = []
i = 0
with gzip.open(CDS_METADATA_PATH, 'rt') as f:
    for l in f:
        json_obj = json.loads(l.strip())

        if (json_obj.get('asin') in cds_ids) :
            data.append(json_obj) 
            i += 1
        
cds_metadata = pd.DataFrame.from_dict(data)

print(f"Loaded CDS metadata: {i} / {len(cds_ids)}")


print("\n\n-- Started to load MOVIES metadata --")
data = []
i = 0
with gzip.open(MOVIES_METADATA_PATH, 'rt') as f:
    for l in f:
        json_obj = json.loads(l.strip())

        if (json_obj.get('asin') in movies_ids) :
            data.append(json_obj) 
            i += 1
        
movies_metadata = pd.DataFrame.from_dict(data)

print(f"Loaded MOVIES metadata: {i} / {len(movies_ids)}")

# -------------------
# Pick only needed metadata
# -------------------
columns_to_consider = ["title", "asin", "brand", "category"]

books_metadata = books_metadata[columns_to_consider]
cds_metadata = cds_metadata[columns_to_consider]
movies_metadata = movies_metadata[columns_to_consider]

books_metadata.to_csv("/home/rinaspata/data/metadata_books.csv")
cds_metadata.to_csv("/home/rinaspata/data/metadata_cds.csv")
movies_metadata.to_csv("/home/rinaspata/data/metadata_movies.csv")

print("Extracted metadata")

import pandas as pd


BOOKS_PATH = "/home/rinaspata/data/full_data/books.csv"
CDS_PATH = "/home/rinaspata/data/full_data/cds.csv"
MOVIES_PATH = "/home/rinaspata/data/full_data/movies.csv"

ITEM_MAPPING_PATH = "/home/rinaspata/data/item_mapping.csv"
USER_MAPPING_PATH = "/home/rinaspata/data/user_mapping.csv"


# -------------------
# Load datasets and ids mapping
# -------------------
print("Loading datasets...")
cds = pd.read_csv(CDS_PATH)
books = pd.read_csv(BOOKS_PATH)
movies = pd.read_csv(MOVIES_PATH)

print("Loading mappings...")
item_mapping = pd.read_csv(ITEM_MAPPING_PATH)
user_mapping = pd.read_csv(USER_MAPPING_PATH)


# -------------------
# Remove rows with non-valid values
# -------------------
columns_to_consider = ["title", "category"]

print("Dropping rows with null values in title or category...")
books.dropna(subset=columns_to_consider, inplace=True)
cds.dropna(subset=columns_to_consider, inplace=True)
movies.dropna(subset=columns_to_consider, inplace=True)


# -------------------
# Pick only valid values
# -------------------
valid_items = set(item_mapping['asin'].unique())
valid_users = set(user_mapping['user'].unique())

print("Filtering data with only valid ones...")
filtered_books = books[books['item_id'].isin(valid_items) & books['user_id'].isin(valid_users)]
filtered_cds = cds[cds['item_id'].isin(valid_items) & cds['user_id'].isin(valid_users)]
filtered_movies = movies[movies['item_id'].isin(valid_items) & movies['user_id'].isin(valid_users)]


# -------------------
# Save results
# -------------------
print("Saving results...")
filtered_books.to_csv("/home/rinaspata/data/full_data/filtered/filtered_books.csv")
filtered_cds.to_csv("/home/rinaspata/data/full_data/filtered/filtered_cds.csv")
filtered_movies.to_csv("/home/rinaspata/data/full_data/filtered/filtered_movies.csv")

print("End.")

import pandas as pd

BOOKS_PATH = "/home/rinaspata/data/amazon/amazon/ratings_Books.csv"
CDS_PATH = "/home/rinaspata/data/amazon/amazon/ratings_CDs_and_Vinyl.csv"
MOVIES_PATH = "/home/rinaspata/data/amazon/amazon/ratings_Movies_and_TV.csv"

BOOKS_METADATA_PATH = "/home/rinaspata/data/metadata_books.csv"
CDS_METADATA_PATH = "/home/rinaspata/data/metadata_cds.csv"
MOVIES_METADATA_PATH = "/home/rinaspata/data/metadata_movies.csv"

COLUMN_HEADERS = ["user_id", "item_id", "rating", "timestamp"]

def merge_data(df_ratings, df_metadata):
    df_metadata = df_metadata.drop(df_metadata.columns[0], axis = 1)
    result = pd.merge(df_ratings, df_metadata, left_on='item_id', right_on='asin', how='left', suffixes=('_ratings', '_metadata'))
    result.drop('asin', axis=1, inplace=True)

    return result

print("Books..")
ratings_books = pd.read_csv(BOOKS_PATH, names=COLUMN_HEADERS)
metadata_books = pd.read_csv(BOOKS_METADATA_PATH)
books_data = merge_data(ratings_books, metadata_books)
books_data.to_csv("/home/rinaspata/data/full_data/books.csv", index=False)
print("Ended books.")

print("Cds..")
ratings_cds = pd.read_csv(CDS_PATH, names=COLUMN_HEADERS)
metadata_cds = pd.read_csv(CDS_METADATA_PATH)
cds_data = merge_data(ratings_cds, metadata_cds)
cds_data.to_csv("/home/rinaspata/data/full_data/cds.csv", index=False)
print("Ended Cds.")

print("Movies..")
ratings_movies = pd.read_csv(MOVIES_PATH, names=COLUMN_HEADERS)
metadata_movies = pd.read_csv(MOVIES_METADATA_PATH)
movies_data = merge_data(ratings_movies, metadata_movies)
movies_data.to_csv("/home/rinaspata/data/full_data/movies.csv", index=False)
print("Ended movies.")

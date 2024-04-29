"""Prompts and related stuff to use among the other scripts"""

# ======================================================================================================================
#                                                       PROMPTS 
# ======================================================================================================================
# Number of tokens to use based on the chosen prompt's lenght
SHORT_PROMPT_TOKENS = 4000
MEDIUM_PROMPT_TOKENS = 5000
LONG_PROMPT_TOKENS = 6000

# Thresholds
LIKED_THRESHOLD = 4
'''
- both lists are in the format = both of them are in the following format
- (if available) = (optional)
'''
# System Prompt
'''
SYSTEM_PROMPT = "You're a cross-domain recommender system. \n\
User provides you a list of items from a base domain that they liked and a list of items such base domain that they disliked \
Both lists are in the format \n\
    1. \n\
        Title: <item's title> \n\
        Brand: <item's brand (optional)> \n\
        Categories: <list of item's categories> \n\
    2. \n\
        Title: <item's title> \n\
        Brand: <item's brand (optional)> \n\
        Categories: <list of item's categories> \n\
    3. \n\
        ... \n\
User also provides a list of candidate items from a target domain in the format \n\
    1. \n\
        Id: <item's id> \n\
        Title: <item's title> \n\
        Brand: <item's brand (optional)> \n\
        Categories: <list of item's categories> \n\
    2. \n\
        Id: <item's id> \n\
        Title: <item's title> \n\
        Brand: <item's brand (optional)> \n\
        Categories: <list of item's categories> \n\
    3. \n\
        ... \n\
You must re-rank the candidate items based on the user's liked items and return the top %d %s"
'''
SYSTEM_PROMPT = "You're a cross-domain recommender system \n\
User provides you a list of items from a base domain that they liked and a list of items such base domain that they disliked \
Using the same format as the user, you must re-rank the candidate items based on the user's liked items and return the top %d %s"
# Explanation Prompts
'''brief suggestion explanation = provide a brief explanation for your suggestion
stesso per explanation suggestion ma senza brief'''
SYSTEM_EXPLANATION_PROMPT = "\nProvide a suggestion explanation"
SYSTEM_SHORT_EXPLANATION_PROMPT = "\nProvide a brief suggestion explanation"
'''explanation = your explanation
answer using the following format = your answer must be in the following format'''
# Output answer format prompt
SYSTEM_ANSWER_FORMAT = " Answer using the format: \n\
' Items ranking: <list of top %d recommended %s' ids>.\n \
 Explanation: <explanation>. '\n"

'''i liked the following = i liked
stessa per le altre'''
# User Prompts
USER_PROMPT = "I liked %s : \n\
%s \n\
Instead, I disliked %s : \n\
%s \n\
Rank and return the top %d %s from: \n\
%s"


# ONLY for No Candidate Items Scenario - per il momento non usato
USER_PROMPT_NO_SUGGESTIONS = "I liked the following %s : \n\
%s \n\
Instead, I disliked the following %s : \n\
%s \n\
Return the top %d %s that you would suggest me"

# Assistant Prompts
ASSISTANT_PROMPT = "Items ranking: %s.\n \
 Explanation: <your explanation>. " 
ASSISTANT_RERANKING_PROMPT = "Based on your likings, I re-rank the %s as:\n %s"  # list of suggested items's ids
print(len(SYSTEM_PROMPT + USER_PROMPT + ASSISTANT_PROMPT + SYSTEM_ANSWER_FORMAT))

# ======================================================================================================================
#                                                      TEST CASES 
# ======================================================================================================================
DATA_DIR = "../../data/processed/"
EXTRA_DATA_DIR = "../../data/processed/extra_cut/"

BOOKS2_5_DIR = DATA_DIR + "Books2_5/"
BOOKS3_5_DIR = DATA_DIR + "Books3_5/"
EXTRA_BOOKS_DIR = EXTRA_DATA_DIR + "Books/"

CDS2_5_DIR = DATA_DIR + "CDs2_5/"
CDS3_5_DIR = DATA_DIR + "CDs3_5/"
EXTRA_CDS_DIR = EXTRA_DATA_DIR + "CDs/"

MOVIES2_5_DIR = DATA_DIR + "Movies2_5/"
MOVIES3_5_DIR = DATA_DIR + "Movies3_5/"
EXTRA_MOVIES_DIR = EXTRA_DATA_DIR + "Movies/"

MOVIES5_10_DIR = EXTRA_DATA_DIR + "Movies_5_10/"
BOOKS5_10_DIR = EXTRA_DATA_DIR + "Books_5_10/"
CDS5_10_DIR = EXTRA_DATA_DIR + "Cds_5_10/"

MOVIES5_20_DIR = EXTRA_DATA_DIR + "Movies_5_20/"
BOOKS5_20_DIR = EXTRA_DATA_DIR + "Books_5_20/"
CDS5_20_DIR = EXTRA_DATA_DIR + "Cds_5_20/"

MOVIES5_30_DIR = EXTRA_DATA_DIR + "Movies_5_30/"
BOOKS5_30_DIR = EXTRA_DATA_DIR + "Books_5_30/"
CDS5_30_DIR = EXTRA_DATA_DIR + "Cds_5_30/"

MOVIES10_20_DIR = EXTRA_DATA_DIR + "Movies_10_20/"
BOOKS10_20_DIR = EXTRA_DATA_DIR + "Books_10_20/"
CDS10_20_DIR = EXTRA_DATA_DIR + "Cds_10_20/"

MOVIES10_30_DIR = EXTRA_DATA_DIR + "Movies_10_30/"
BOOKS10_30_DIR = EXTRA_DATA_DIR + "Books_10_30/"
CDS10_30_DIR = EXTRA_DATA_DIR + "Cds_10_30/"

TARGET_DIR = EXTRA_DATA_DIR + "Targets/"

OUTPUT_DIRECTORY = "models/predictions/logs/"
RESULTS_DIRECTORY = "models/predictions/"

COLUMNS_TYPES =  {
        'user_id': 'string',
        'item_id': 'string',
        'rating': 'string',
        'timestamp': 'string',
        'title': 'string',
        'brand': 'string',
        'category': 'string'
    }

# M_1/2 = B20/30 | M_3/4 = C20/30 | B_1/2 = M20/30 | B_3/4 = C20_30 | C_1/2 = B20/30 | C_3/4 = M20_30

TEST_CASES = [   
    "movies530_books1020", "movies530_books1030",                                          # 52,53
    "movies530_cds1020", "movies530_cds1030",                                           # 54,55
    "books530_movies1020", "books530_movies1030",                                           # 56,57
    "books530_cds1020", "books530_cds1030",                                           # 58,59
    "cds530_books520", "cds530_books1030",                                           # 60,61
    "cds530_movies520", "cds530_movies1030",                                            # 62,63
    
]


TEST_CASES_DICT = {
    "movies530_books1030": [MOVIES5_30_DIR + "movies_5_30_2", TARGET_DIR + "books_10_30"], 
    "movies530_cds1030": [MOVIES5_30_DIR + "movies_5_30_4", TARGET_DIR + "cds_10_30"],
    "books530_movies1030": [BOOKS5_30_DIR + "books_5_30_2", TARGET_DIR + "movies_10_30"], 
    "books530_cds1030": [BOOKS5_30_DIR + "books_5_30_4", TARGET_DIR + "cds_10_30"], 
    "cds530_books1030": [CDS5_30_DIR + "cds_5_30_2", TARGET_DIR + "books_10_30"],
    "cds530_movies1030": [CDS5_30_DIR + "cds_5_30_4", TARGET_DIR + "movies_10_30"], 
}       

MAX_SUGGESTION = 0
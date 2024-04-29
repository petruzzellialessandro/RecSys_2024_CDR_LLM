from transformers import GPT2Tokenizer
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


tokenizer = GPT2Tokenizer.from_pretrained("openai/chatgpt")
text = SYSTEM_PROMPT+SYSTEM_EXPLANATION_PROMPT+SYSTEM_ANSWER_FORMAT+USER_PROMPT+ASSISTANT_PROMPT
token_count = len(tokenizer.encode(text, add_special_tokens=True))

print(f"Numero di token per il testo: {token_count}")
import pandas as pd
import os
import random
import conf as conf
import re


def data_loader(data_paths):
    """
        Method to load datasets given their paths
    """
    liked_items_data = pd.read_csv(data_paths[0] + ".csv", dtype=conf.COLUMNS_TYPES)
    candidate_items_data = pd.read_csv(data_paths[1] + ".csv", dtype=conf.COLUMNS_TYPES)
    print("Datasets loaded!")
    return liked_items_data, candidate_items_data


def get_user_id(user_ids, start_index, i):
    """
        Gets the user id from the user ids list, starting from the start index and incrementing it by i.
    """
    j = start_index + i

    if j >= len(user_ids):
        j = 0
    
    return user_ids[j], j


def get_sample_user_id(user_ids, sample_user_ids, start_index):
    """
        Gets a random user id from the user ids list, excluding the ones in the sample user ids list.
    """
    user_ids_length = len(user_ids)
    random_index = start_index

    while random_index == start_index:
        random_index = random.randint(0, user_ids_length - 1)

        if user_ids[random_index] in sample_user_ids:
            random_index = start_index

    return user_ids[random_index]


def get_liked_items(user_id, base_items, is_target_domain=False):
    """
        Gets the liked items for a user from the base items DataFrame.
        If is_target_domain is True, the liked items are shuffled.
    """
    liked_items_for_user = base_items[base_items["user_id"] == user_id]
    
    if is_target_domain:
        liked_items_for_user = liked_items_for_user.sample(frac=1).reset_index(drop=True)

    return liked_items_for_user


def sort_liked_items(user_liked_items, no_correct_items=5):
    """
        Sorts the liked items for a user by rating and timestamp.
    """
    sorted_user_liked_items = user_liked_items.sort_values(by=['rating', 'timestamp'], ascending=[False, True])

    return sorted_user_liked_items.head(no_correct_items)


def get_candidate_items(user_id, target_items, n_correct_items=3, max_suggestions=7):
    """
        Gets the candidate items for a user from the target items DataFrame.
        The candidate items are composed of the correct items (liked by the user) and wrong items (not liked by the user).
    """
    candidate_items = target_items[target_items["user_id"] == user_id]
    correct_item_ids = candidate_items["item_id"].head(n_correct_items).tolist()

    n_wrong_items = max_suggestions - n_correct_items
    wrong_items_df = target_items[target_items["user_id"] != user_id]
    wrong_items_df = wrong_items_df.drop_duplicates(subset=["item_id"])

    candidate_items = pd.concat([candidate_items[:n_correct_items], wrong_items_df[:n_wrong_items]], ignore_index=True)
    candidate_items = candidate_items.sample(frac=1).reset_index(drop=True)
    
    return candidate_items, correct_item_ids


def convert_rating(rating, threshold=4):
    """
        Converts a rating to 'liked' or 'disliked' according to a threshold.
    """
    if int(rating) >= threshold:
        return 'liked'
    else:
        return 'disliked'


def split_items_for_user(items):
    """
        Splits the items for a user into liked and disliked items.
    """
    liked_items = items[items["rating"].astype(int) > conf.LIKED_THRESHOLD]
    disliked_items = items[items["rating"].astype(int) <= conf.LIKED_THRESHOLD]

    return liked_items, disliked_items


def format_items_list(is_liked_items_list, item_list, is_free_assistant=False):
    """
        Formats a list of items to be displayed in the prompt.
    """
    formatted = ""
    i = 0

    for _, row in item_list.iterrows():
        i += 1

        formatted += str(i) + ".\n\t"

        if not is_liked_items_list and not is_free_assistant:
            formatted += "Id: " + row["item_id"] + "\n\t"

        formatted += "Title: " + row["title"] + "\n\t"

        if pd.notna(row["brand"]):
            formatted += "Brand: " + row["brand"] + "\n\t"
        
        formatted += "Categories: " + row["category"] + "\n\t"

    return formatted


def filter_llm_output(output_message, model):
    """
        Filters the output message of the LLM to get the recommendation and the explanation.
    """
    filtered = _get_llm_answer(output_message, model)

    prediction, explanation = _extract_ranking_and_explanation(filtered)

    return prediction, explanation


def _get_llm_answer(output_message, model):
    """
        Gets the answer of the LLM from its output message.
    """
    if model == "mistral":
        temp_split = output_message.split('</s>')
        final_split = temp_split[-2].strip().split('[/INST]') 
        filtered = final_split[-1].strip()
    
    elif model == "gpt":
        filtered = output_message

    return filtered


def _extract_ranking_and_explanation(answer):
    """
        Extracts the item ranking and the explanation from the answer of the LLM.
    """
    
    sentences = answer.split('.\n')
    
    # Extract the item ranking
    ranking_match = re.search(r"Items ranking: ?\[?([^\]]+)\]?", sentences[0])
    item_ranking = ranking_match.group(1).split(', ') if ranking_match else []
    item_ranking = [item.strip("'") for item in item_ranking]

    # Extract the explanation
    if len(sentences) > 1 and sentences[1]:
        explanation_match = re.search(r"Explanation: (.+)", sentences[1])
        explanation = explanation_match.group(1).strip() if explanation_match else ""
    else:
        print(f"===========\nNO EXPLANATION FOUND!!!!!!!!!!!!!!\n===========\n")
        explanation = answer

    return item_ranking, explanation


def build_messages(k, system_prompt, sample_user_prompts, assistant_prompts, user_prompt, is_mistral=False):
    """
        Builds the messages to be sent to the LLM.
    """
    messages = []

    if not is_mistral:
        messages.append({"role": "system", "content": system_prompt})

    for i in range(k):
        messages.append({"role": "user", "content": sample_user_prompts[i][0]})
        messages.append({"role": "assistant", "content": assistant_prompts[i][0]})

    messages.append({"role": "user", "content": user_prompt})

    return messages


def add_prediction(df, user_id, ground_truth, prediction, explanation):
    """
        Adds a prediction to the DataFrame.
    """
    new_row = pd.DataFrame({'UserId': [user_id], 'TrueRanking': [ground_truth], 'PredictedRanking': [prediction], 'Explanation': [explanation]})

    df = pd.concat([df, new_row], ignore_index=True)

    return df


def save_df_to_folder(df, folder_path, file_name):
    """
        Saves a DataFrame to a folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, file_name)

    df.to_pickle(file_path)

    print(f"DataFrame saved to {file_path}")

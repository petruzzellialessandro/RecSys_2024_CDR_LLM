import os
import pandas as pd
import time
from ScenarioGenerator import ScenarioGenerator
import utilities as ut
import conf as conf
import openai
import argparse
from tqdm import tqdm


# -----------------
# Global variables
# -----------------
MODEL = "gpt-3.5-turbo"
openai.api_key = "key-open-AI"

# -----------------
# Main function
# -----------------
def main(args):
    print("==== START ====")
    print(args)

    BASE_DOMAIN = args.base_domain
    TARGET_DOMAIN = args.target_domain
    data_scenario = args.test_case
    BATCH_SIZE = int(args.batch_size)

    print(f"\nSelected test case: {data_scenario}!")
    data_paths_to_load = conf.TEST_CASES_DICT.get(data_scenario)
    print(f"The following is to be loaded:\n"
      f"- Liked items: {data_paths_to_load[0]}\n"
      f"- Candidate items: {data_paths_to_load[1]}\n"
      )
    #print('1')
    liked_items, candidate_items = ut.data_loader(data_paths_to_load)
    #print('2')

    BASE_DOMAIN_DATA = liked_items
    TARGET_DOMAIN_DATA = candidate_items

    user_ids = BASE_DOMAIN_DATA["user_id"].unique().tolist()
    #print('4')

    OUTPUT_DIR = conf.OUTPUT_DIRECTORY + data_scenario + "/"
    RESULT_DIR = conf.RESULTS_DIRECTORY + data_scenario + "/"
    #print(RESULT_DIR)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # ---------------------
    # Zero-shot scenario
    # ---------------------
    
    print("\n\n--- Zero-shot scenario ---")
    scenario = ScenarioGenerator(scenario="zero-shot")
    k = 0

    LOG_FILE_NAME = "GPT_0s_yE_" + str(conf.MAX_SUGGESTION) + "sug_" + str(conf.LONG_PROMPT_TOKENS) + "TK.txt"
    LOG_OUTPUT_FILE_PATH = OUTPUT_DIR + LOG_FILE_NAME
    
    gpt_0s = get_gpt_recs_per_batches(user_ids, BATCH_SIZE, scenario, k, BASE_DOMAIN, BASE_DOMAIN_DATA,
                                      TARGET_DOMAIN, TARGET_DOMAIN_DATA, conf.MAX_SUGGESTION, LOG_OUTPUT_FILE_PATH)
    
    RESULTS_FILE_NAME = data_scenario + "_GPT_0s.pkl"
    ut.save_df_to_folder(gpt_0s, RESULT_DIR, RESULTS_FILE_NAME)

    # ---------------------
    # One-shot scenario
    # ---------------------
    print("\n\n--- One-shot scenario ---")
    scenario = ScenarioGenerator(scenario="one-shot")
    k = 1

    LOG_FILE_NAME = "GPT_1s_yE_" + str(conf.MAX_SUGGESTION) + "sug_" + str(conf.LONG_PROMPT_TOKENS) + "TK.txt"
    LOG_OUTPUT_FILE_PATH = OUTPUT_DIR + LOG_FILE_NAME

    gpt_1s = get_gpt_recs_per_batches(user_ids, BATCH_SIZE, scenario, k, BASE_DOMAIN, BASE_DOMAIN_DATA,
                                      TARGET_DOMAIN, TARGET_DOMAIN_DATA, conf.MAX_SUGGESTION, LOG_OUTPUT_FILE_PATH)
    
    RESULTS_FILE_NAME = data_scenario + "_GPT_1s.pkl"
    ut.save_df_to_folder(gpt_1s, RESULT_DIR, RESULTS_FILE_NAME)

    print("==== END ====")


# -----------------
# Helper functions
# -----------------
def get_gpt_recs_per_batches(user_ids, batch_size, scenario, k, base_domain, base_domain_data,
                    target_domain, target_domain_data, max_suggestions, log_file_path):
    """
        Processes the user ids in batches.
    """
    num_batches = (len(user_ids) + batch_size - 1) // batch_size

    final_df = pd.DataFrame(columns=['UserId', 'TrueRanking', 'PredictedRanking', 'Explanation'])

    for i in range(num_batches):
        # Calculate start and end indices for the current batch
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size

        current_batch = user_ids[start_idx:end_idx]
        batch_df = gpt_recsys(scenario, k, base_domain, base_domain_data, target_domain,
                            target_domain_data, max_suggestions, current_batch, log_file_path)
        final_df = pd.concat([final_df, batch_df], ignore_index=True)

        time.sleep(15)  # sleep for 15 seconds to avoid errors on OpenAI's server

    return final_df




def gpt_recsys(scenario, k, base_dom, base_dom_data, target_dom, target_dom_data, 
             no_max_sugg, id_users_list, out_file_path, is_explaining=True):
    """
        Generates the prompts for GPT and gets its predictions.
    """
    print(f"Test Scenario: {scenario.scenario}\n")
    
    results_df = pd.DataFrame(columns=['UserId', 'TrueRanking', 'PredictedRanking', 'Explanation'])
    spent_secs_per_request = []
 
    try:
        with open(out_file_path, "a") as output_file: 
        
            for i in tqdm(range(len(id_users_list))):
                # Record the start time
                start_time = time.time()
    
    
                # Generate prompts
                system_prompt, sample_user_prompts, assistant_prompts, user_prompt, ground_truth = scenario.generate_scenario(output_file=output_file, 
                                                                                            start_index=i, 
                                                                                            user_ids=id_users_list, 
                                                                                            has_explanation=is_explaining,
                                                                                            base_domain=base_dom, 
                                                                                            base_domain_data=base_dom_data, 
                                                                                            target_domain=target_dom, 
                                                                                            target_domain_data=target_dom_data,
                                                                                            max_suggestions=no_max_sugg)
                output_file.write(f"\nsystem_prompt: \n{system_prompt}")
                output_file.write(f"\nuser_prompt: \n{user_prompt}")
                # Build messages and get LLM's response
                messages = ut.build_messages(k, system_prompt, sample_user_prompts, assistant_prompts, user_prompt)
                #time.sleep(20) testing
                response = openai.ChatCompletion.create(
                    model = MODEL,
                    messages=messages,
                    temperature=0,
                )

                prediction, explanation = ut.filter_llm_output(response.choices[0]['message']['content'], "gpt")
                output_file.write(f"\nLLM answer: \n{response.choices[0]['message']['content']}")
                output_file.write("\n-------------------")
    
                results_df = ut.add_prediction(results_df, id_users_list[i], ground_truth, prediction, explanation)
    
                # Record the end time
                end_time = time.time()
    
                # Calculate and print the time spent
                time_spent = end_time - start_time
                spent_secs_per_request.append(time_spent)
                time.sleep(5)  # sleep for 2 seconds to avoid exceeding the OpenAI API rate limit or other kind of errors
                    
    except Exception as e:
        print(f"Error: {e}")
        print(f"Processed users: {len(results_df)}")

    print("Test ended!")

    if len(spent_secs_per_request) > 0:
        print(f"Average time spent per request: {sum(spent_secs_per_request) / len(spent_secs_per_request)} secs")
        print(f"Total spent time: {sum(spent_secs_per_request)} secs")
    else:
        print("No available data about the time spent per request.")
   
    return results_df





if __name__ == '__main__':
    # Parsing necessary arguments for the desidered test case
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--base-domain', type=str, help='Name of the base domain')
    aparser.add_argument('--target-domain', type=str, help='Name of the target domain')
    aparser.add_argument('--test-case', type=str, help='Name of the test case in the dictionary')
    aparser.add_argument('--batch-size', type=str, help='Number of users per batch')
    
    main(aparser.parse_args())
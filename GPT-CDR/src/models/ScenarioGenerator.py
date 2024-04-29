import time
from PromptGenerator import PromptGenerator
import utilities as ut

class ScenarioGenerator:

    def __init__(self, scenario, is_free=False, full_log=True):

        self.scenario = scenario
        self.system_prompt_generator = PromptGenerator("system")
        self.user_prompt_generator = PromptGenerator("user")
        if is_free:
            self.assistant_prompt_generator = PromptGenerator("free_assistant")
        else:
            self.assistant_prompt_generator = PromptGenerator("assistant")
        self.full_log = full_log


    def generate_k_shot_scenario(self, k, output_file, start_index, user_ids, 
                                 has_explanation=False, has_suggest=True,
                                 base_domain="", base_domain_data=[], 
                                 target_domain="", target_domain_data=[],
                                 no_item_to_suggest=5, is_mistral=False):
        """
            Generate prompts for a k-shot scenario.

            Parameters:
            - k (int): The number of shots in the scenario.
            - output_file (file): File where prompt metadata are stored.
            - start_index (int): The starting index for retrieving user data.
            - user_ids (list): List of user identifiers.
            - has_explanation (bool): Whether the prompts should include explanations.
            - has_suggest (bool): Whether the user prompt should include a list of candidate items from which the LLM should recommend
            - base_domain (str): The name of the base domain.
            - base_domain_data (list): Data related to the base domain from which picking user's preferences.
            - target_domain (str): The name of the target domain.
            - target_domain_data (list): Data related to the target domain from which picking items for recommendations.
            - no_liked_items (int): Number of liked items to consider for each user.
            - no_suggestions (int): Number of candidate items for each user.
            - max_suggestions (int): Maximum number of suggestions the LLM has to give.
            - correct_items (int): Number of correct items in the candidate list.

            Returns:
            - system prompt (str): Prompt for the system role.
            - sample user prompts (list<str>): List of prompts for the sample users role (used only in few-shots scenarios).
            - assistant prompts (list<str>): List of prompts for the assistants role (used only in few-shots scenarios).
            - user prompt (str): Prompt for the test user.
            - ground truth (list<str>): List of ranked items for the test user.
        """

        sample_users_ids = []
        all_sample_user_prompts = []
        all_assistant_prompts = []

        

        output_file.write("\n=================================================================")
        output_file.write(f"\n{k}-shot scenario")
        if has_explanation:
            output_file.write("\nYes explanations")
        else:
            output_file.write("\nNo explanations")
        if has_suggest:
            output_file.write(" - Yes suggestions")
        else:
            output_file.write(" - No suggestions")
        output_file.write("\n=================================================================")


        # -------------------
        # SYSTEM PROMPT
        # -------------------
        system_prompt = self.system_prompt_generator.generate_prompt(has_explanation=has_explanation,
                                                            target_domain=target_domain,
                                                            max_suggestions=no_item_to_suggest)

        # -------------------
        # SAMPLE USER AND ASSISTANT PROMPTS
        # -------------------
        for _ in range(k):
            # Get sample user's id
            user_id_sample = ut.get_sample_user_id(user_ids, sample_users_ids, start_index)

            # Liked items - BASE DOMAIN
            liked_items_for_user_sample = ut.get_liked_items(user_id_sample, base_domain_data)
            # Liked items - TARGET DOMAIN
            candidate_items_for_user_sample = ut.get_liked_items(user_id_sample, target_domain_data, is_target_domain=True)
            # Sort the candidate items by Rating and Timestamp
            sorted_candidate_items_sample = ut.sort_liked_items(candidate_items_for_user_sample)
            
            sample_users_ids.append(user_id_sample)

            if self.full_log:
                output_file.write("\n-------------------")
                output_file.write(f"\nSample user id: {user_id_sample}")
                output_file.write("\nLiked items in the base domain:\n")
                output_file.write(liked_items_for_user_sample.to_string())

                output_file.write("\nCandidate items in the target domain:\n")
                output_file.write(candidate_items_for_user_sample.to_string())

                output_file.write("\nCorrect items for Sample user:\n")
                output_file.write(sorted_candidate_items_sample['item_id'].to_string())

            # Build sample user and assistant prompts
            sample_user_prompt = self.user_prompt_generator.generate_prompt(has_explanation=has_explanation, 
                                                                            has_suggest=has_suggest, 
                                                                            base_domain=base_domain,
                                                                            target_domain=target_domain,
                                                                            max_suggestions=no_item_to_suggest,
                                                                            liked_items=liked_items_for_user_sample,
                                                                            candidate_items=candidate_items_for_user_sample,
                                                                            is_mistral_used=is_mistral)

            assistant_prompt = self.assistant_prompt_generator.generate_prompt(target_domain=target_domain,
                                                                               suggested_items=sorted_candidate_items_sample)

            all_sample_user_prompts.append([sample_user_prompt])
            all_assistant_prompts.append([assistant_prompt])


        # -------------------
        # TEST USER PROMPT
        # -------------------
        # Get data for test user
        user_id_test = user_ids[start_index]

        # Liked items - BASE DOMAIN
        liked_items_for_user_test = ut.get_liked_items(user_id_test, base_domain_data)
        # Liked items - TARGET DOMAIN
        candidate_items_for_user_test = ut.get_liked_items(user_id_test, target_domain_data, is_target_domain=True)
        
        # Sort the candidate items by Rating and Timestamp
        sorted_candidate_items_test = ut.sort_liked_items(candidate_items_for_user_test)
        ground_truth = sorted_candidate_items_test['item_id'].tolist()
        output_file.write("\n\n-------------------")
        output_file.write(f"\nTest user id: {user_id_test}")

        if self.full_log:
            output_file.write("\nLiked items in the base domain:\n")
            output_file.write(liked_items_for_user_test.to_string())

            output_file.write("\nCandidate items in the target domain:\n")
            output_file.write(candidate_items_for_user_test.to_string())

        output_file.write("\nCorrect items for test user:\n")
        output_file.write(ground_truth.__str__())
        
        # Build test user prompt
        user_prompt = self.user_prompt_generator.generate_prompt(has_explanation=has_explanation, 
                                                                 has_suggest=has_suggest, 
                                                                 base_domain=base_domain,
                                                                 target_domain=target_domain,
                                                                 max_suggestions=no_item_to_suggest,
                                                                 liked_items=liked_items_for_user_test,
                                                                 candidate_items=candidate_items_for_user_test,
                                                                 is_mistral_used=is_mistral)

        return system_prompt, all_sample_user_prompts, all_assistant_prompts, user_prompt, ground_truth


    def generate_scenario(self, output_file, start_index, user_ids,
                          has_explanation=False, has_suggest=True,
                          base_domain="", base_domain_data=[],
                          target_domain="", target_domain_data=[],
                          max_suggestions=5):
        """
            Generate prompts for a given scenario, incorporating k-shot variations.

            Parameters:
            - scenario (str): The scenario type. It can be: "zero-shot", "one-shot", or "three-shot".
            - start_index (int): The starting index for retrieving user data.
            - user_ids (list): List of user identifiers.
            - has_explanation (bool): Whether the prompts should include explanations.
            - has_suggest (bool): Whether the user prompt should include a list of candidate items from which the LLM should recommend
            - base_domain (str): The name of the base domain.
            - base_domain_data (list): Data related to the base domain from which picking user's preferences.
            - target_domain (str): The name of the target domain.
            - target_domain_data (list): Data related to the target domain from which picking items for recommendations.
            - max_suggestions (int): Maximum number of suggestions the LLM has to give.

            Returns:
            - system prompt (str): Prompt for the system role.
            - sample user prompts (list<str>): List of prompts for the sample users role (used only in few-shots scenarios).
            - assistant prompts (list<str>): List of prompts for the assistants role (used only in few-shots scenarios).
            - user prompt (str): Prompt for the test user.
        """

        sample_user_prompts = []
        assistant_prompts = []

        if self.scenario == "zero-shot":
            k = 0

        elif self.scenario == "one-shot":
            k = 1

        elif self.scenario == "three-shot":
            k = 3

        system_prompt, sample_user_prompts, assistant_prompts, user_prompt, ground_truth = self.generate_k_shot_scenario(
            k, output_file, start_index, user_ids, 
            has_explanation, has_suggest, 
            base_domain, base_domain_data, 
            target_domain, target_domain_data,
            no_item_to_suggest=max_suggestions
            )

        return system_prompt, sample_user_prompts, assistant_prompts, user_prompt, ground_truth
